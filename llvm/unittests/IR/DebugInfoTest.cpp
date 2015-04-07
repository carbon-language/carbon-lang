//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DebugNodeTest, getFlag) {
  // Some valid flags.
  EXPECT_EQ(DebugNode::FlagPublic, DebugNode::getFlag("DIFlagPublic"));
  EXPECT_EQ(DebugNode::FlagProtected, DebugNode::getFlag("DIFlagProtected"));
  EXPECT_EQ(DebugNode::FlagPrivate, DebugNode::getFlag("DIFlagPrivate"));
  EXPECT_EQ(DebugNode::FlagVector, DebugNode::getFlag("DIFlagVector"));
  EXPECT_EQ(DebugNode::FlagRValueReference,
            DebugNode::getFlag("DIFlagRValueReference"));

  // FlagAccessibility shouldn't work.
  EXPECT_EQ(0u, DebugNode::getFlag("DIFlagAccessibility"));

  // Some other invalid strings.
  EXPECT_EQ(0u, DebugNode::getFlag("FlagVector"));
  EXPECT_EQ(0u, DebugNode::getFlag("Vector"));
  EXPECT_EQ(0u, DebugNode::getFlag("other things"));
  EXPECT_EQ(0u, DebugNode::getFlag("DIFlagOther"));
}

TEST(DebugNodeTest, getFlagString) {
  // Some valid flags.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DebugNode::getFlagString(DebugNode::FlagPublic));
  EXPECT_EQ(StringRef("DIFlagProtected"),
            DebugNode::getFlagString(DebugNode::FlagProtected));
  EXPECT_EQ(StringRef("DIFlagPrivate"),
            DebugNode::getFlagString(DebugNode::FlagPrivate));
  EXPECT_EQ(StringRef("DIFlagVector"),
            DebugNode::getFlagString(DebugNode::FlagVector));
  EXPECT_EQ(StringRef("DIFlagRValueReference"),
            DebugNode::getFlagString(DebugNode::FlagRValueReference));

  // FlagAccessibility actually equals FlagPublic.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DebugNode::getFlagString(DebugNode::FlagAccessibility));

  // Some other invalid flags.
  EXPECT_EQ(StringRef(), DebugNode::getFlagString(DebugNode::FlagPublic |
                                                  DebugNode::FlagVector));
  EXPECT_EQ(StringRef(), DebugNode::getFlagString(DebugNode::FlagFwdDecl |
                                                  DebugNode::FlagArtificial));
  EXPECT_EQ(StringRef(), DebugNode::getFlagString(0xffff));
}

TEST(DebugNodeTest, splitFlags) {
// Some valid flags.
#define CHECK_SPLIT(FLAGS, VECTOR, REMAINDER)                                  \
  {                                                                            \
    SmallVector<unsigned, 8> V;                                                \
    EXPECT_EQ(REMAINDER, DebugNode::splitFlags(FLAGS, V));                     \
    EXPECT_TRUE(makeArrayRef(V).equals(VECTOR));                               \
  }
  CHECK_SPLIT(DebugNode::FlagPublic, {DebugNode::FlagPublic}, 0u);
  CHECK_SPLIT(DebugNode::FlagProtected, {DebugNode::FlagProtected}, 0u);
  CHECK_SPLIT(DebugNode::FlagPrivate, {DebugNode::FlagPrivate}, 0u);
  CHECK_SPLIT(DebugNode::FlagVector, {DebugNode::FlagVector}, 0u);
  CHECK_SPLIT(DebugNode::FlagRValueReference, {DebugNode::FlagRValueReference},
              0u);
  unsigned Flags[] = {DebugNode::FlagFwdDecl, DebugNode::FlagVector};
  CHECK_SPLIT(DebugNode::FlagFwdDecl | DebugNode::FlagVector, Flags, 0u);
  CHECK_SPLIT(0x100000u, {}, 0x100000u);
  CHECK_SPLIT(0x100000u | DebugNode::FlagVector, {DebugNode::FlagVector},
              0x100000u);
#undef CHECK_SPLIT
}

} // end namespace
