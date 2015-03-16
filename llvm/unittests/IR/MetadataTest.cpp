//===- unittests/IR/MetadataTest.cpp - Metadata unit tests ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
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
  ContextAndReplaceableUses CRU(make_unique<ReplaceableMetadataImpl>(Context));
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_TRUE(CRU.hasReplaceableUses());
  EXPECT_TRUE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, makeReplaceable) {
  LLVMContext Context;
  ContextAndReplaceableUses CRU(Context);
  CRU.makeReplaceable(make_unique<ReplaceableMetadataImpl>(Context));
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_TRUE(CRU.hasReplaceableUses());
  EXPECT_TRUE(CRU.getReplaceableUses());
}

TEST(ContextAndReplaceableUsesTest, takeReplaceableUses) {
  LLVMContext Context;
  auto ReplaceableUses = make_unique<ReplaceableMetadataImpl>(Context);
  auto *Ptr = ReplaceableUses.get();
  ContextAndReplaceableUses CRU(std::move(ReplaceableUses));
  ReplaceableUses = CRU.takeReplaceableUses();
  EXPECT_EQ(&Context, &CRU.getContext());
  EXPECT_FALSE(CRU.hasReplaceableUses());
  EXPECT_FALSE(CRU.getReplaceableUses());
  EXPECT_EQ(Ptr, ReplaceableUses.get());
}

class MetadataTest : public testing::Test {
protected:
  LLVMContext Context;
  MDNode *getNode() { return MDNode::get(Context, None); }
  MDNode *getNode(Metadata *MD) { return MDNode::get(Context, MD); }
  MDNode *getNode(Metadata *MD1, Metadata *MD2) {
    Metadata *MDs[] = {MD1, MD2};
    return MDNode::get(Context, MDs);
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
  char *str = new char[13];
  strncpy(str, "testing 1 2 3", 13);
  MDString *s = MDString::get(Context, StringRef(str, 13));
  strncpy(str, "aaaaaaaaaaaaa", 13);
  delete[] str;

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
  EXPECT_STREQ("!\"\\00\\0A\\22\\5C\\FF\"", oss.str().c_str());
}

typedef MetadataTest MDNodeTest;

// Test the two constructors, and containing other Constants.
TEST_F(MDNodeTest, Simple) {
  char x[3] = { 'a', 'b', 'c' };
  char y[3] = { '1', '2', '3' };

  MDString *s1 = MDString::get(Context, StringRef(&x[0], 3));
  MDString *s2 = MDString::get(Context, StringRef(&y[0], 3));
  ConstantAsMetadata *CI = ConstantAsMetadata::get(
      ConstantInt::get(getGlobalContext(), APInt(8, 0)));

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
  Constant *C = ConstantInt::get(Type::getInt32Ty(getGlobalContext()), 1);
  Instruction *I = new BitCastInst(C, Type::getInt32Ty(getGlobalContext()));

  Metadata *const V = LocalAsMetadata::get(I);
  MDNode *n = MDNode::get(Context, V);
  TrackingMDRef wvh(n);

  EXPECT_EQ(n, wvh);

  delete I;
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

typedef MetadataTest MDLocationTest;

TEST_F(MDLocationTest, Overflow) {
  MDNode *N = MDNode::get(Context, None);
  {
    MDLocation *L = MDLocation::get(Context, 2, 7, N);
    EXPECT_EQ(2u, L->getLine());
    EXPECT_EQ(7u, L->getColumn());
  }
  unsigned U16 = 1u << 16;
  {
    MDLocation *L = MDLocation::get(Context, UINT32_MAX, U16 - 1, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(U16 - 1, L->getColumn());
  }
  {
    MDLocation *L = MDLocation::get(Context, UINT32_MAX, U16, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(0u, L->getColumn());
  }
  {
    MDLocation *L = MDLocation::get(Context, UINT32_MAX, U16 + 1, N);
    EXPECT_EQ(UINT32_MAX, L->getLine());
    EXPECT_EQ(0u, L->getColumn());
  }
}

TEST_F(MDLocationTest, getDistinct) {
  MDNode *N = MDNode::get(Context, None);
  MDLocation *L0 = MDLocation::getDistinct(Context, 2, 7, N);
  EXPECT_TRUE(L0->isDistinct());
  MDLocation *L1 = MDLocation::get(Context, 2, 7, N);
  EXPECT_FALSE(L1->isDistinct());
  EXPECT_EQ(L1, MDLocation::get(Context, 2, 7, N));
}

TEST_F(MDLocationTest, getTemporary) {
  MDNode *N = MDNode::get(Context, None);
  auto L = MDLocation::getTemporary(Context, 2, 7, N);
  EXPECT_TRUE(L->isTemporary());
  EXPECT_FALSE(L->isResolved());
}

typedef MetadataTest GenericDebugNodeTest;

TEST_F(GenericDebugNodeTest, get) {
  StringRef Header = "header";
  auto *Empty = MDNode::get(Context, None);
  Metadata *Ops1[] = {Empty};
  auto *N = GenericDebugNode::get(Context, 15, Header, Ops1);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(2u, N->getNumOperands());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(MDString::get(Context, Header), N->getOperand(0));
  EXPECT_EQ(1u, N->getNumDwarfOperands());
  EXPECT_EQ(Empty, N->getDwarfOperand(0));
  EXPECT_EQ(Empty, N->getOperand(1));
  ASSERT_TRUE(N->isUniqued());

  EXPECT_EQ(N, GenericDebugNode::get(Context, 15, Header, Ops1));

  N->replaceOperandWith(1, nullptr);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(nullptr, N->getDwarfOperand(0));
  ASSERT_TRUE(N->isUniqued());

  Metadata *Ops2[] = {nullptr};
  EXPECT_EQ(N, GenericDebugNode::get(Context, 15, Header, Ops2));

  N->replaceDwarfOperandWith(0, Empty);
  EXPECT_EQ(15u, N->getTag());
  EXPECT_EQ(Header, N->getHeader());
  EXPECT_EQ(Empty, N->getDwarfOperand(0));
  ASSERT_TRUE(N->isUniqued());
  EXPECT_EQ(N, GenericDebugNode::get(Context, 15, Header, Ops1));

  TempGenericDebugNode Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(GenericDebugNodeTest, getEmptyHeader) {
  // Canonicalize !"" to null.
  auto *N = GenericDebugNode::get(Context, 15, StringRef(), None);
  EXPECT_EQ(StringRef(), N->getHeader());
  EXPECT_EQ(nullptr, N->getOperand(0));
}

typedef MetadataTest MDSubrangeTest;

TEST_F(MDSubrangeTest, get) {
  auto *N = MDSubrange::get(Context, 5, 7);
  EXPECT_EQ(dwarf::DW_TAG_subrange_type, N->getTag());
  EXPECT_EQ(5, N->getCount());
  EXPECT_EQ(7, N->getLo());
  EXPECT_EQ(N, MDSubrange::get(Context, 5, 7));
  EXPECT_EQ(MDSubrange::get(Context, 5, 0), MDSubrange::get(Context, 5));

  TempMDSubrange Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDSubrangeTest, getEmptyArray) {
  auto *N = MDSubrange::get(Context, -1, 0);
  EXPECT_EQ(dwarf::DW_TAG_subrange_type, N->getTag());
  EXPECT_EQ(-1, N->getCount());
  EXPECT_EQ(0, N->getLo());
  EXPECT_EQ(N, MDSubrange::get(Context, -1, 0));
}

typedef MetadataTest MDEnumeratorTest;

TEST_F(MDEnumeratorTest, get) {
  auto *N = MDEnumerator::get(Context, 7, "name");
  EXPECT_EQ(dwarf::DW_TAG_enumerator, N->getTag());
  EXPECT_EQ(7, N->getValue());
  EXPECT_EQ("name", N->getName());
  EXPECT_EQ(N, MDEnumerator::get(Context, 7, "name"));

  EXPECT_NE(N, MDEnumerator::get(Context, 8, "name"));
  EXPECT_NE(N, MDEnumerator::get(Context, 7, "nam"));

  TempMDEnumerator Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDBasicTypeTest;

TEST_F(MDBasicTypeTest, get) {
  auto *N =
      MDBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33, 26, 7);
  EXPECT_EQ(dwarf::DW_TAG_base_type, N->getTag());
  EXPECT_EQ("special", N->getName());
  EXPECT_EQ(33u, N->getSizeInBits());
  EXPECT_EQ(26u, N->getAlignInBits());
  EXPECT_EQ(7u, N->getEncoding());
  EXPECT_EQ(0u, N->getLine());
  EXPECT_EQ(N, MDBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 7));

  EXPECT_NE(N, MDBasicType::get(Context, dwarf::DW_TAG_unspecified_type,
                                "special", 33, 26, 7));
  EXPECT_NE(N,
            MDBasicType::get(Context, dwarf::DW_TAG_base_type, "s", 33, 26, 7));
  EXPECT_NE(N, MDBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 32,
                                26, 7));
  EXPECT_NE(N, MDBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                25, 7));
  EXPECT_NE(N, MDBasicType::get(Context, dwarf::DW_TAG_base_type, "special", 33,
                                26, 6));

  TempMDBasicType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDBasicTypeTest, getWithLargeValues) {
  auto *N = MDBasicType::get(Context, dwarf::DW_TAG_base_type, "special",
                             UINT64_MAX, UINT64_MAX - 1, 7);
  EXPECT_EQ(UINT64_MAX, N->getSizeInBits());
  EXPECT_EQ(UINT64_MAX - 1, N->getAlignInBits());
}

TEST_F(MDBasicTypeTest, getUnspecified) {
  auto *N =
      MDBasicType::get(Context, dwarf::DW_TAG_unspecified_type, "unspecified");
  EXPECT_EQ(dwarf::DW_TAG_unspecified_type, N->getTag());
  EXPECT_EQ("unspecified", N->getName());
  EXPECT_EQ(0u, N->getSizeInBits());
  EXPECT_EQ(0u, N->getAlignInBits());
  EXPECT_EQ(0u, N->getEncoding());
  EXPECT_EQ(0u, N->getLine());
}

typedef MetadataTest MDTypeTest;

TEST_F(MDTypeTest, clone) {
  // Check that MDType has a specialized clone that returns TempMDType.
  MDType *N = MDBasicType::get(Context, dwarf::DW_TAG_base_type, "int", 32, 32,
                               dwarf::DW_ATE_signed);

  TempMDType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDTypeTest, setFlags) {
  // void (void)
  Metadata *TypesOps[] = {nullptr};
  Metadata *Types = MDTuple::get(Context, TypesOps);

  MDType *D = MDSubroutineType::getDistinct(Context, 0u, Types);
  EXPECT_EQ(0u, D->getFlags());
  D->setFlags(DIDescriptor::FlagRValueReference);
  EXPECT_EQ(DIDescriptor::FlagRValueReference, D->getFlags());
  D->setFlags(0u);
  EXPECT_EQ(0u, D->getFlags());

  TempMDType T = MDSubroutineType::getTemporary(Context, 0u, Types);
  EXPECT_EQ(0u, T->getFlags());
  T->setFlags(DIDescriptor::FlagRValueReference);
  EXPECT_EQ(DIDescriptor::FlagRValueReference, T->getFlags());
  T->setFlags(0u);
  EXPECT_EQ(0u, T->getFlags());
}

typedef MetadataTest MDDerivedTypeTest;

TEST_F(MDDerivedTypeTest, get) {
  Metadata *File = MDTuple::getDistinct(Context, None);
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *BaseType = MDTuple::getDistinct(Context, None);
  Metadata *ExtraData = MDTuple::getDistinct(Context, None);

  auto *N = MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, Scope, BaseType, 2, 3, 4, 5, ExtraData);
  EXPECT_EQ(dwarf::DW_TAG_pointer_type, N->getTag());
  EXPECT_EQ("something", N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(1u, N->getLine());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(BaseType, N->getBaseType());
  EXPECT_EQ(2u, N->getSizeInBits());
  EXPECT_EQ(3u, N->getAlignInBits());
  EXPECT_EQ(4u, N->getOffsetInBits());
  EXPECT_EQ(5u, N->getFlags());
  EXPECT_EQ(ExtraData, N->getExtraData());
  EXPECT_EQ(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));

  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_reference_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "else",
                                  File, 1, Scope, BaseType, 2, 3, 4, 5,
                                  ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", Scope, 1, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 2, Scope, BaseType, 2, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N,
            MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, File, BaseType, 2, 3, 4, 5, ExtraData));
  EXPECT_NE(N,
            MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, Scope, File, 2, 3, 4, 5, ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 3, 3,
                                  4, 5, ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 2,
                                  4, 5, ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  5, 5, ExtraData));
  EXPECT_NE(N, MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type,
                                  "something", File, 1, Scope, BaseType, 2, 3,
                                  4, 4, ExtraData));
  EXPECT_NE(N,
            MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, Scope, BaseType, 2, 3, 4, 5, File));

  TempMDDerivedType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDDerivedTypeTest, getWithLargeValues) {
  Metadata *File = MDTuple::getDistinct(Context, None);
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *BaseType = MDTuple::getDistinct(Context, None);
  Metadata *ExtraData = MDTuple::getDistinct(Context, None);

  auto *N = MDDerivedType::get(Context, dwarf::DW_TAG_pointer_type, "something",
                               File, 1, Scope, BaseType, UINT64_MAX,
                               UINT64_MAX - 1, UINT64_MAX - 2, 5, ExtraData);
  EXPECT_EQ(UINT64_MAX, N->getSizeInBits());
  EXPECT_EQ(UINT64_MAX - 1, N->getAlignInBits());
  EXPECT_EQ(UINT64_MAX - 2, N->getOffsetInBits());
}

typedef MetadataTest MDCompositeTypeTest;

TEST_F(MDCompositeTypeTest, get) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 1;
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *BaseType = MDTuple::getDistinct(Context, None);
  uint64_t SizeInBits = 2;
  uint64_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  unsigned Flags = 5;
  Metadata *Elements = MDTuple::getDistinct(Context, None);
  unsigned RuntimeLang = 6;
  Metadata *VTableHolder = MDTuple::getDistinct(Context, None);
  Metadata *TemplateParams = MDTuple::getDistinct(Context, None);
  StringRef Identifier = "some id";

  auto *N = MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
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
  EXPECT_EQ(Elements, N->getElements());
  EXPECT_EQ(RuntimeLang, N->getRuntimeLang());
  EXPECT_EQ(VTableHolder, N->getVTableHolder());
  EXPECT_EQ(TemplateParams, N->getTemplateParams());
  EXPECT_EQ(Identifier, N->getIdentifier());

  EXPECT_EQ(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));

  EXPECT_NE(N, MDCompositeType::get(Context, Tag + 1, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, "abc", File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, Scope, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line + 1, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, File,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope, File,
                                    SizeInBits, AlignInBits, OffsetInBits,
                                    Flags, Elements, RuntimeLang, VTableHolder,
                                    TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits + 1, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits + 1,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits + 1, Flags, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags + 1, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, File, RuntimeLang,
                                    VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang + 1,
                   VTableHolder, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    File, TemplateParams, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, File, Identifier));
  EXPECT_NE(N, MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                    BaseType, SizeInBits, AlignInBits,
                                    OffsetInBits, Flags, Elements, RuntimeLang,
                                    VTableHolder, TemplateParams, "other"));

  // Be sure that missing identifiers get null pointers.
  EXPECT_FALSE(MDCompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                   VTableHolder, TemplateParams, "")->getRawIdentifier());
  EXPECT_FALSE(MDCompositeType::get(
                   Context, Tag, Name, File, Line, Scope, BaseType, SizeInBits,
                   AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
                   VTableHolder, TemplateParams)->getRawIdentifier());

  TempMDCompositeType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDCompositeTypeTest, getWithLargeValues) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 1;
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *BaseType = MDTuple::getDistinct(Context, None);
  uint64_t SizeInBits = UINT64_MAX;
  uint64_t AlignInBits = UINT64_MAX - 1;
  uint64_t OffsetInBits = UINT64_MAX - 2;
  unsigned Flags = 5;
  Metadata *Elements = MDTuple::getDistinct(Context, None);
  unsigned RuntimeLang = 6;
  Metadata *VTableHolder = MDTuple::getDistinct(Context, None);
  Metadata *TemplateParams = MDTuple::getDistinct(Context, None);
  StringRef Identifier = "some id";

  auto *N = MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                 BaseType, SizeInBits, AlignInBits,
                                 OffsetInBits, Flags, Elements, RuntimeLang,
                                 VTableHolder, TemplateParams, Identifier);
  EXPECT_EQ(SizeInBits, N->getSizeInBits());
  EXPECT_EQ(AlignInBits, N->getAlignInBits());
  EXPECT_EQ(OffsetInBits, N->getOffsetInBits());
}

TEST_F(MDCompositeTypeTest, replaceOperands) {
  unsigned Tag = dwarf::DW_TAG_structure_type;
  StringRef Name = "some name";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 1;
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *BaseType = MDTuple::getDistinct(Context, None);
  uint64_t SizeInBits = 2;
  uint64_t AlignInBits = 3;
  uint64_t OffsetInBits = 4;
  unsigned Flags = 5;
  unsigned RuntimeLang = 6;
  StringRef Identifier = "some id";

  auto *N = MDCompositeType::get(Context, Tag, Name, File, Line, Scope,
                                 BaseType, SizeInBits, AlignInBits,
                                 OffsetInBits, Flags, nullptr, RuntimeLang,
                                 nullptr, nullptr, Identifier);

  auto *Elements = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getElements());
  N->replaceElements(Elements);
  EXPECT_EQ(Elements, N->getElements());
  N->replaceElements(nullptr);
  EXPECT_EQ(nullptr, N->getElements());

  auto *VTableHolder = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getVTableHolder());
  N->replaceVTableHolder(VTableHolder);
  EXPECT_EQ(VTableHolder, N->getVTableHolder());
  N->replaceVTableHolder(nullptr);
  EXPECT_EQ(nullptr, N->getVTableHolder());

  auto *TemplateParams = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getTemplateParams());
  N->replaceTemplateParams(TemplateParams);
  EXPECT_EQ(TemplateParams, N->getTemplateParams());
  N->replaceTemplateParams(nullptr);
  EXPECT_EQ(nullptr, N->getTemplateParams());
}

typedef MetadataTest MDSubroutineTypeTest;

TEST_F(MDSubroutineTypeTest, get) {
  unsigned Flags = 1;
  Metadata *TypeArray = MDTuple::getDistinct(Context, None);

  auto *N = MDSubroutineType::get(Context, Flags, TypeArray);
  EXPECT_EQ(dwarf::DW_TAG_subroutine_type, N->getTag());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(TypeArray, N->getTypeArray());
  EXPECT_EQ(N, MDSubroutineType::get(Context, Flags, TypeArray));

  EXPECT_NE(N, MDSubroutineType::get(Context, Flags + 1, TypeArray));
  EXPECT_NE(N, MDSubroutineType::get(Context, Flags,
                                     MDTuple::getDistinct(Context, None)));

  TempMDSubroutineType Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));

  // Test always-empty operands.
  EXPECT_EQ(nullptr, N->getScope());
  EXPECT_EQ(nullptr, N->getFile());
  EXPECT_EQ("", N->getName());
  EXPECT_EQ(nullptr, N->getBaseType());
  EXPECT_EQ(nullptr, N->getVTableHolder());
  EXPECT_EQ(nullptr, N->getTemplateParams());
  EXPECT_EQ("", N->getIdentifier());
}

typedef MetadataTest MDFileTest;

TEST_F(MDFileTest, get) {
  StringRef Filename = "file";
  StringRef Directory = "dir";
  auto *N = MDFile::get(Context, Filename, Directory);

  EXPECT_EQ(dwarf::DW_TAG_file_type, N->getTag());
  EXPECT_EQ(Filename, N->getFilename());
  EXPECT_EQ(Directory, N->getDirectory());
  EXPECT_EQ(N, MDFile::get(Context, Filename, Directory));

  EXPECT_NE(N, MDFile::get(Context, "other", Directory));
  EXPECT_NE(N, MDFile::get(Context, Filename, "other"));

  TempMDFile Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDFileTest, ScopeGetFile) {
  // Ensure that MDScope::getFile() returns itself.
  MDScope *N = MDFile::get(Context, "file", "dir");
  EXPECT_EQ(N, N->getFile());
}

typedef MetadataTest MDCompileUnitTest;

TEST_F(MDCompileUnitTest, get) {
  unsigned SourceLanguage = 1;
  Metadata *File = MDTuple::getDistinct(Context, None);
  StringRef Producer = "some producer";
  bool IsOptimized = false;
  StringRef Flags = "flag after flag";
  unsigned RuntimeVersion = 2;
  StringRef SplitDebugFilename = "another/file";
  unsigned EmissionKind = 3;
  Metadata *EnumTypes = MDTuple::getDistinct(Context, None);
  Metadata *RetainedTypes = MDTuple::getDistinct(Context, None);
  Metadata *Subprograms = MDTuple::getDistinct(Context, None);
  Metadata *GlobalVariables = MDTuple::getDistinct(Context, None);
  Metadata *ImportedEntities = MDTuple::getDistinct(Context, None);
  auto *N = MDCompileUnit::get(
      Context, SourceLanguage, File, Producer, IsOptimized, Flags,
      RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
      RetainedTypes, Subprograms, GlobalVariables, ImportedEntities);

  EXPECT_EQ(dwarf::DW_TAG_compile_unit, N->getTag());
  EXPECT_EQ(SourceLanguage, N->getSourceLanguage());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Producer, N->getProducer());
  EXPECT_EQ(IsOptimized, N->isOptimized());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(RuntimeVersion, N->getRuntimeVersion());
  EXPECT_EQ(SplitDebugFilename, N->getSplitDebugFilename());
  EXPECT_EQ(EmissionKind, N->getEmissionKind());
  EXPECT_EQ(EnumTypes, N->getEnumTypes());
  EXPECT_EQ(RetainedTypes, N->getRetainedTypes());
  EXPECT_EQ(Subprograms, N->getSubprograms());
  EXPECT_EQ(GlobalVariables, N->getGlobalVariables());
  EXPECT_EQ(ImportedEntities, N->getImportedEntities());
  EXPECT_EQ(N, MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                                  IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));

  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage + 1, File, Producer,
                                  IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, EnumTypes, Producer,
                                  IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, File, "other",
                                  IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                                  !IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                                  IsOptimized, "other", RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                                  IsOptimized, Flags, RuntimeVersion + 1,
                                  SplitDebugFilename, EmissionKind, EnumTypes,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N,
            MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                               IsOptimized, Flags, RuntimeVersion, "other",
                               EmissionKind, EnumTypes, RetainedTypes,
                               Subprograms, GlobalVariables, ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                                  IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind + 1,
                                  EnumTypes, RetainedTypes, Subprograms,
                                  GlobalVariables, ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(Context, SourceLanguage, File, Producer,
                                  IsOptimized, Flags, RuntimeVersion,
                                  SplitDebugFilename, EmissionKind, File,
                                  RetainedTypes, Subprograms, GlobalVariables,
                                  ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(
                   Context, SourceLanguage, File, Producer, IsOptimized, Flags,
                   RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
                   File, Subprograms, GlobalVariables, ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(
                   Context, SourceLanguage, File, Producer, IsOptimized, Flags,
                   RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
                   RetainedTypes, File, GlobalVariables, ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(
                   Context, SourceLanguage, File, Producer, IsOptimized, Flags,
                   RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
                   RetainedTypes, Subprograms, File, ImportedEntities));
  EXPECT_NE(N, MDCompileUnit::get(
                   Context, SourceLanguage, File, Producer, IsOptimized, Flags,
                   RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
                   RetainedTypes, Subprograms, GlobalVariables, File));

  TempMDCompileUnit Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDCompileUnitTest, replaceArrays) {
  unsigned SourceLanguage = 1;
  Metadata *File = MDTuple::getDistinct(Context, None);
  StringRef Producer = "some producer";
  bool IsOptimized = false;
  StringRef Flags = "flag after flag";
  unsigned RuntimeVersion = 2;
  StringRef SplitDebugFilename = "another/file";
  unsigned EmissionKind = 3;
  Metadata *EnumTypes = MDTuple::getDistinct(Context, None);
  Metadata *RetainedTypes = MDTuple::getDistinct(Context, None);
  Metadata *ImportedEntities = MDTuple::getDistinct(Context, None);
  auto *N = MDCompileUnit::get(
      Context, SourceLanguage, File, Producer, IsOptimized, Flags,
      RuntimeVersion, SplitDebugFilename, EmissionKind, EnumTypes,
      RetainedTypes, nullptr, nullptr, ImportedEntities);

  auto *Subprograms = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getSubprograms());
  N->replaceSubprograms(Subprograms);
  EXPECT_EQ(Subprograms, N->getSubprograms());
  N->replaceSubprograms(nullptr);
  EXPECT_EQ(nullptr, N->getSubprograms());

  auto *GlobalVariables = MDTuple::getDistinct(Context, None);
  EXPECT_EQ(nullptr, N->getGlobalVariables());
  N->replaceGlobalVariables(GlobalVariables);
  EXPECT_EQ(GlobalVariables, N->getGlobalVariables());
  N->replaceGlobalVariables(nullptr);
  EXPECT_EQ(nullptr, N->getGlobalVariables());
}

typedef MetadataTest MDSubprogramTest;

TEST_F(MDSubprogramTest, get) {
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 2;
  Metadata *Type = MDTuple::getDistinct(Context, None);
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  unsigned ScopeLine = 3;
  Metadata *ContainingType = MDTuple::getDistinct(Context, None);
  unsigned Virtuality = 4;
  unsigned VirtualIndex = 5;
  unsigned Flags = 6;
  bool IsOptimized = false;
  Metadata *Function = MDTuple::getDistinct(Context, None);
  Metadata *TemplateParams = MDTuple::getDistinct(Context, None);
  Metadata *Declaration = MDTuple::getDistinct(Context, None);
  Metadata *Variables = MDTuple::getDistinct(Context, None);

  auto *N = MDSubprogram::get(
      Context, Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
      IsDefinition, ScopeLine, ContainingType, Virtuality, VirtualIndex, Flags,
      IsOptimized, Function, TemplateParams, Declaration, Variables);

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
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(IsOptimized, N->isOptimized());
  EXPECT_EQ(Function, N->getFunction());
  EXPECT_EQ(TemplateParams, N->getTemplateParams());
  EXPECT_EQ(Declaration, N->getDeclaration());
  EXPECT_EQ(Variables, N->getVariables());
  EXPECT_EQ(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));

  EXPECT_NE(N, MDSubprogram::get(Context, File, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, "other", LinkageName, File,
                                 Line, Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized, Function,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, "other", File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, Scope, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File,
                                 Line + 1, Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized, Function,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Scope, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, !IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, !IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition,
                                 ScopeLine + 1, ContainingType, Virtuality,
                                 VirtualIndex, Flags, IsOptimized, Function,
                                 TemplateParams, Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 Type, Virtuality, VirtualIndex, Flags,
                                 IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality + 1, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex + 1,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 ~Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, !IsOptimized, Function, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Type, TemplateParams,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, Type,
                                 Declaration, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Type, Variables));
  EXPECT_NE(N, MDSubprogram::get(Context, Scope, Name, LinkageName, File, Line,
                                 Type, IsLocalToUnit, IsDefinition, ScopeLine,
                                 ContainingType, Virtuality, VirtualIndex,
                                 Flags, IsOptimized, Function, TemplateParams,
                                 Declaration, Type));

  TempMDSubprogram Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDSubprogramTest, replaceFunction) {
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 2;
  Metadata *Type = MDTuple::getDistinct(Context, None);
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  unsigned ScopeLine = 3;
  Metadata *ContainingType = MDTuple::getDistinct(Context, None);
  unsigned Virtuality = 4;
  unsigned VirtualIndex = 5;
  unsigned Flags = 6;
  bool IsOptimized = false;
  Metadata *TemplateParams = MDTuple::getDistinct(Context, None);
  Metadata *Declaration = MDTuple::getDistinct(Context, None);
  Metadata *Variables = MDTuple::getDistinct(Context, None);

  auto *N = MDSubprogram::get(
      Context, Scope, Name, LinkageName, File, Line, Type, IsLocalToUnit,
      IsDefinition, ScopeLine, ContainingType, Virtuality, VirtualIndex, Flags,
      IsOptimized, nullptr, TemplateParams, Declaration, Variables);

  EXPECT_EQ(nullptr, N->getFunction());

  std::unique_ptr<Function> F(
      Function::Create(FunctionType::get(Type::getVoidTy(Context), false),
                       GlobalValue::ExternalLinkage));
  N->replaceFunction(F.get());
  EXPECT_EQ(ConstantAsMetadata::get(F.get()), N->getFunction());

  N->replaceFunction(nullptr);
  EXPECT_EQ(nullptr, N->getFunction());
}

typedef MetadataTest MDLexicalBlockTest;

TEST_F(MDLexicalBlockTest, get) {
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 5;
  unsigned Column = 8;

  auto *N = MDLexicalBlock::get(Context, Scope, File, Line, Column);

  EXPECT_EQ(dwarf::DW_TAG_lexical_block, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Column, N->getColumn());
  EXPECT_EQ(N, MDLexicalBlock::get(Context, Scope, File, Line, Column));

  EXPECT_NE(N, MDLexicalBlock::get(Context, File, File, Line, Column));
  EXPECT_NE(N, MDLexicalBlock::get(Context, Scope, Scope, Line, Column));
  EXPECT_NE(N, MDLexicalBlock::get(Context, Scope, File, Line + 1, Column));
  EXPECT_NE(N, MDLexicalBlock::get(Context, Scope, File, Line, Column + 1));

  TempMDLexicalBlock Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDLexicalBlockFileTest;

TEST_F(MDLexicalBlockFileTest, get) {
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Discriminator = 5;

  auto *N = MDLexicalBlockFile::get(Context, Scope, File, Discriminator);

  EXPECT_EQ(dwarf::DW_TAG_lexical_block, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Discriminator, N->getDiscriminator());
  EXPECT_EQ(N, MDLexicalBlockFile::get(Context, Scope, File, Discriminator));

  EXPECT_NE(N, MDLexicalBlockFile::get(Context, File, File, Discriminator));
  EXPECT_NE(N, MDLexicalBlockFile::get(Context, Scope, Scope, Discriminator));
  EXPECT_NE(N,
            MDLexicalBlockFile::get(Context, Scope, File, Discriminator + 1));

  TempMDLexicalBlockFile Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDNamespaceTest;

TEST_F(MDNamespaceTest, get) {
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *File = MDTuple::getDistinct(Context, None);
  StringRef Name = "namespace";
  unsigned Line = 5;

  auto *N = MDNamespace::get(Context, Scope, File, Name, Line);

  EXPECT_EQ(dwarf::DW_TAG_namespace, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(N, MDNamespace::get(Context, Scope, File, Name, Line));

  EXPECT_NE(N, MDNamespace::get(Context, File, File, Name, Line));
  EXPECT_NE(N, MDNamespace::get(Context, Scope, Scope, Name, Line));
  EXPECT_NE(N, MDNamespace::get(Context, Scope, File, "other", Line));
  EXPECT_NE(N, MDNamespace::get(Context, Scope, File, Name, Line + 1));

  TempMDNamespace Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDTemplateTypeParameterTest;

TEST_F(MDTemplateTypeParameterTest, get) {
  StringRef Name = "template";
  Metadata *Type = MDTuple::getDistinct(Context, None);
  Metadata *Other = MDTuple::getDistinct(Context, None);

  auto *N = MDTemplateTypeParameter::get(Context, Name, Type);

  EXPECT_EQ(dwarf::DW_TAG_template_type_parameter, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(N, MDTemplateTypeParameter::get(Context, Name, Type));

  EXPECT_NE(N, MDTemplateTypeParameter::get(Context, "other", Type));
  EXPECT_NE(N, MDTemplateTypeParameter::get(Context, Name, Other));

  TempMDTemplateTypeParameter Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDTemplateValueParameterTest;

TEST_F(MDTemplateValueParameterTest, get) {
  unsigned Tag = dwarf::DW_TAG_template_value_parameter;
  StringRef Name = "template";
  Metadata *Type = MDTuple::getDistinct(Context, None);
  Metadata *Value = MDTuple::getDistinct(Context, None);
  Metadata *Other = MDTuple::getDistinct(Context, None);

  auto *N = MDTemplateValueParameter::get(Context, Tag, Name, Type, Value);
  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(Value, N->getValue());
  EXPECT_EQ(N, MDTemplateValueParameter::get(Context, Tag, Name, Type, Value));

  EXPECT_NE(N, MDTemplateValueParameter::get(
                   Context, dwarf::DW_TAG_GNU_template_template_param, Name,
                   Type, Value));
  EXPECT_NE(N, MDTemplateValueParameter::get(Context, Tag,  "other", Type,
                                             Value));
  EXPECT_NE(N, MDTemplateValueParameter::get(Context, Tag,  Name, Other,
                                             Value));
  EXPECT_NE(N, MDTemplateValueParameter::get(Context, Tag, Name, Type, Other));

  TempMDTemplateValueParameter Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDGlobalVariableTest;

TEST_F(MDGlobalVariableTest, get) {
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  StringRef Name = "name";
  StringRef LinkageName = "linkage";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 5;
  Metadata *Type = MDTuple::getDistinct(Context, None);
  bool IsLocalToUnit = false;
  bool IsDefinition = true;
  Metadata *Variable = MDTuple::getDistinct(Context, None);
  Metadata *StaticDataMemberDeclaration = MDTuple::getDistinct(Context, None);

  auto *N = MDGlobalVariable::get(Context, Scope, Name, LinkageName, File, Line,
                                  Type, IsLocalToUnit, IsDefinition, Variable,
                                  StaticDataMemberDeclaration);
  EXPECT_EQ(dwarf::DW_TAG_variable, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(LinkageName, N->getLinkageName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(IsLocalToUnit, N->isLocalToUnit());
  EXPECT_EQ(IsDefinition, N->isDefinition());
  EXPECT_EQ(Variable, N->getVariable());
  EXPECT_EQ(StaticDataMemberDeclaration, N->getStaticDataMemberDeclaration());
  EXPECT_EQ(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));

  EXPECT_NE(N, MDGlobalVariable::get(Context, File, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, "other", LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, "other", File, Line,
                                     Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, Scope,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N,
            MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                  Line + 1, Type, IsLocalToUnit, IsDefinition,
                                  Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Scope, IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, !IsLocalToUnit, IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, !IsDefinition,
                                     Variable, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Type, StaticDataMemberDeclaration));
  EXPECT_NE(N, MDGlobalVariable::get(Context, Scope, Name, LinkageName, File,
                                     Line, Type, IsLocalToUnit, IsDefinition,
                                     Variable, Type));

  TempMDGlobalVariable Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDLocalVariableTest;

TEST_F(MDLocalVariableTest, get) {
  unsigned Tag = dwarf::DW_TAG_arg_variable;
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  StringRef Name = "name";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 5;
  Metadata *Type = MDTuple::getDistinct(Context, None);
  unsigned Arg = 6;
  unsigned Flags = 7;
  Metadata *InlinedAtScope = MDTuple::getDistinct(Context, None);
  Metadata *InlinedAt =
      MDLocation::getDistinct(Context, 10, 20, InlinedAtScope);

  auto *N = MDLocalVariable::get(Context, Tag, Scope, Name, File, Line, Type,
                                 Arg, Flags, InlinedAt);
  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(Arg, N->getArg());
  EXPECT_EQ(Flags, N->getFlags());
  EXPECT_EQ(InlinedAt, N->getInlinedAt());
  EXPECT_EQ(N, MDLocalVariable::get(Context, Tag, Scope, Name, File, Line, Type,
                                    Arg, Flags, InlinedAt));

  EXPECT_NE(N, MDLocalVariable::get(Context, dwarf::DW_TAG_auto_variable, Scope,
                                    Name, File, Line, Type, Arg, Flags,
                                    InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, File, Name, File, Line,
                                    Type, Arg, Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, "other", File, Line,
                                    Type, Arg, Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, Name, Scope, Line,
                                    Type, Arg, Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, Name, File, Line + 1,
                                    Type, Arg, Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, Name, File, Line,
                                    Scope, Arg, Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, Name, File, Line, Type,
                                    Arg + 1, Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, Name, File, Line, Type,
                                    Arg, ~Flags, InlinedAt));
  EXPECT_NE(N, MDLocalVariable::get(Context, Tag, Scope, Name, File, Line, Type,
                                    Arg, Flags, Scope));

  TempMDLocalVariable Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));

  auto *Inlined = N->withoutInline();
  EXPECT_NE(N, Inlined);
  EXPECT_EQ(N->getTag(), Inlined->getTag());
  EXPECT_EQ(N->getScope(), Inlined->getScope());
  EXPECT_EQ(N->getName(), Inlined->getName());
  EXPECT_EQ(N->getFile(), Inlined->getFile());
  EXPECT_EQ(N->getLine(), Inlined->getLine());
  EXPECT_EQ(N->getType(), Inlined->getType());
  EXPECT_EQ(N->getArg(), Inlined->getArg());
  EXPECT_EQ(N->getFlags(), Inlined->getFlags());
  EXPECT_EQ(nullptr, Inlined->getInlinedAt());
  EXPECT_EQ(N, Inlined->withInline(cast<MDLocation>(InlinedAt)));
}

typedef MetadataTest MDExpressionTest;

TEST_F(MDExpressionTest, get) {
  uint64_t Elements[] = {2, 6, 9, 78, 0};
  auto *N = MDExpression::get(Context, Elements);
  EXPECT_EQ(makeArrayRef(Elements), N->getElements());
  EXPECT_EQ(N, MDExpression::get(Context, Elements));

  EXPECT_EQ(5u, N->getNumElements());
  EXPECT_EQ(2u, N->getElement(0));
  EXPECT_EQ(6u, N->getElement(1));
  EXPECT_EQ(9u, N->getElement(2));
  EXPECT_EQ(78u, N->getElement(3));
  EXPECT_EQ(0u, N->getElement(4));

  TempMDExpression Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

TEST_F(MDExpressionTest, isValid) {
#define EXPECT_VALID(...)                                                      \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    EXPECT_TRUE(MDExpression::get(Context, Elements)->isValid());              \
  } while (false)
#define EXPECT_INVALID(...)                                                    \
  do {                                                                         \
    uint64_t Elements[] = {__VA_ARGS__};                                       \
    EXPECT_FALSE(MDExpression::get(Context, Elements)->isValid());             \
  } while (false)

  // Empty expression should be valid.
  EXPECT_TRUE(MDExpression::get(Context, None));

  // Valid constructions.
  EXPECT_VALID(dwarf::DW_OP_plus, 6);
  EXPECT_VALID(dwarf::DW_OP_deref);
  EXPECT_VALID(dwarf::DW_OP_bit_piece, 3, 7);
  EXPECT_VALID(dwarf::DW_OP_plus, 6, dwarf::DW_OP_deref);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_plus, 6);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_bit_piece, 3, 7);
  EXPECT_VALID(dwarf::DW_OP_deref, dwarf::DW_OP_plus, 6, dwarf::DW_OP_bit_piece, 3, 7);

  // Invalid constructions.
  EXPECT_INVALID(~0u);
  EXPECT_INVALID(dwarf::DW_OP_plus);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece, 3);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece, 3, 7, dwarf::DW_OP_plus, 3);
  EXPECT_INVALID(dwarf::DW_OP_bit_piece, 3, 7, dwarf::DW_OP_deref);

#undef EXPECT_VALID
#undef EXPECT_INVALID
}

typedef MetadataTest MDObjCPropertyTest;

TEST_F(MDObjCPropertyTest, get) {
  StringRef Name = "name";
  Metadata *File = MDTuple::getDistinct(Context, None);
  unsigned Line = 5;
  StringRef GetterName = "getter";
  StringRef SetterName = "setter";
  unsigned Attributes = 7;
  Metadata *Type = MDTuple::getDistinct(Context, None);

  auto *N = MDObjCProperty::get(Context, Name, File, Line, GetterName,
                                SetterName, Attributes, Type);

  EXPECT_EQ(dwarf::DW_TAG_APPLE_property, N->getTag());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(File, N->getFile());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(GetterName, N->getGetterName());
  EXPECT_EQ(SetterName, N->getSetterName());
  EXPECT_EQ(Attributes, N->getAttributes());
  EXPECT_EQ(Type, N->getType());
  EXPECT_EQ(N, MDObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes, Type));

  EXPECT_NE(N, MDObjCProperty::get(Context, "other", File, Line, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, MDObjCProperty::get(Context, Name, Type, Line, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, MDObjCProperty::get(Context, Name, File, Line + 1, GetterName,
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, MDObjCProperty::get(Context, Name, File, Line, "other",
                                   SetterName, Attributes, Type));
  EXPECT_NE(N, MDObjCProperty::get(Context, Name, File, Line, GetterName,
                                   "other", Attributes, Type));
  EXPECT_NE(N, MDObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes + 1, Type));
  EXPECT_NE(N, MDObjCProperty::get(Context, Name, File, Line, GetterName,
                                   SetterName, Attributes, File));

  TempMDObjCProperty Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
}

typedef MetadataTest MDImportedEntityTest;

TEST_F(MDImportedEntityTest, get) {
  unsigned Tag = dwarf::DW_TAG_imported_module;
  Metadata *Scope = MDTuple::getDistinct(Context, None);
  Metadata *Entity = MDTuple::getDistinct(Context, None);
  unsigned Line = 5;
  StringRef Name = "name";

  auto *N = MDImportedEntity::get(Context, Tag, Scope, Entity, Line, Name);

  EXPECT_EQ(Tag, N->getTag());
  EXPECT_EQ(Scope, N->getScope());
  EXPECT_EQ(Entity, N->getEntity());
  EXPECT_EQ(Line, N->getLine());
  EXPECT_EQ(Name, N->getName());
  EXPECT_EQ(N, MDImportedEntity::get(Context, Tag, Scope, Entity, Line, Name));

  EXPECT_NE(N,
            MDImportedEntity::get(Context, dwarf::DW_TAG_imported_declaration,
                                  Scope, Entity, Line, Name));
  EXPECT_NE(N, MDImportedEntity::get(Context, Tag, Entity, Entity, Line, Name));
  EXPECT_NE(N, MDImportedEntity::get(Context, Tag, Scope, Scope, Line, Name));
  EXPECT_NE(N,
            MDImportedEntity::get(Context, Tag, Scope, Entity, Line + 1, Name));
  EXPECT_NE(N,
            MDImportedEntity::get(Context, Tag, Scope, Entity, Line, "other"));

  TempMDImportedEntity Temp = N->clone();
  EXPECT_EQ(N, MDNode::replaceWithUniqued(std::move(Temp)));
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

TEST_F(ValueAsMetadataTest, CollidingDoubleUpdates) {
  // Create a constant.
  ConstantAsMetadata *CI = ConstantAsMetadata::get(
      ConstantInt::get(getGlobalContext(), APInt(8, 0)));

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
}
