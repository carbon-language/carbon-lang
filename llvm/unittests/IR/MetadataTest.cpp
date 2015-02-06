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
#include "llvm/IR/DebugInfoMetadata.h"
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
    OS << "!{";
    C->printAsOperand(OS);
    OS << ", ";
    S->printAsOperand(OS);
    OS << ", null";
    MDNode *Nodes[] = {N0, N1, N2};
    for (auto *Node : Nodes)
      OS << ", <" << (void *)Node << ">";
    OS << "}\n";
  }

  std::string Actual;
  {
    raw_string_ostream OS(Actual);
    N->print(OS);
  }

  EXPECT_EQ(Expected, Actual);
}

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
}

TEST_F(GenericDebugNodeTest, getEmptyHeader) {
  // Canonicalize !"" to null.
  auto *N = GenericDebugNode::get(Context, 15, StringRef(), None);
  EXPECT_EQ(StringRef(), N->getHeader());
  EXPECT_EQ(nullptr, N->getOperand(0));
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
