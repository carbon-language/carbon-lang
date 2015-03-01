//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {

static void PrintTo(const StringRef &S, ::std::ostream *os) {
  *os << "(" << (const void *)S.data() << "," << S.size() << ") = '";
  for (auto C : S)
    if (C)
      *os << C;
    else
      *os << "\\00";
  *os << "'";
}
static void PrintTo(const DIHeaderFieldIterator &I, ::std::ostream *os) {
  PrintTo(I.getCurrent(), os);
  *os << " in ";
  PrintTo(I.getHeader(), os);
}

} // end namespace llvm

namespace {

#define MAKE_FIELD_ITERATOR(S)                                                 \
  DIHeaderFieldIterator(StringRef(S, sizeof(S) - 1))
TEST(DebugInfoTest, DIHeaderFieldIterator) {
  ASSERT_EQ(DIHeaderFieldIterator(), DIHeaderFieldIterator());

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR(""));
  ASSERT_EQ(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR(""));
  ASSERT_EQ("", *DIHeaderFieldIterator(""));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("stuff"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("stuff"));
  ASSERT_EQ("stuff", *DIHeaderFieldIterator("stuff"));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_NE(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++++MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_EQ("st", *MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_EQ("uff", *++MAKE_FIELD_ITERATOR("st\0uff"));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_NE(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++++MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_EQ("stuff", *MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_EQ("", *++MAKE_FIELD_ITERATOR("stuff\0"));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_NE(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++++MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_EQ("", *MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_EQ("stuff", *++MAKE_FIELD_ITERATOR("\0stuff"));
}

TEST(DIDescriptorTest, getFlag) {
  // Some valid flags.
  EXPECT_EQ(DIDescriptor::FlagPublic, DIDescriptor::getFlag("DIFlagPublic"));
  EXPECT_EQ(DIDescriptor::FlagProtected,
            DIDescriptor::getFlag("DIFlagProtected"));
  EXPECT_EQ(DIDescriptor::FlagPrivate, DIDescriptor::getFlag("DIFlagPrivate"));
  EXPECT_EQ(DIDescriptor::FlagVector, DIDescriptor::getFlag("DIFlagVector"));
  EXPECT_EQ(DIDescriptor::FlagRValueReference,
            DIDescriptor::getFlag("DIFlagRValueReference"));

  // FlagAccessibility shouldn't work.
  EXPECT_EQ(0u, DIDescriptor::getFlag("DIFlagAccessibility"));

  // Some other invalid strings.
  EXPECT_EQ(0u, DIDescriptor::getFlag("FlagVector"));
  EXPECT_EQ(0u, DIDescriptor::getFlag("Vector"));
  EXPECT_EQ(0u, DIDescriptor::getFlag("other things"));
  EXPECT_EQ(0u, DIDescriptor::getFlag("DIFlagOther"));
}

TEST(DIDescriptorTest, getFlagString) {
  // Some valid flags.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DIDescriptor::getFlagString(DIDescriptor::FlagPublic));
  EXPECT_EQ(StringRef("DIFlagProtected"),
            DIDescriptor::getFlagString(DIDescriptor::FlagProtected));
  EXPECT_EQ(StringRef("DIFlagPrivate"),
            DIDescriptor::getFlagString(DIDescriptor::FlagPrivate));
  EXPECT_EQ(StringRef("DIFlagVector"),
            DIDescriptor::getFlagString(DIDescriptor::FlagVector));
  EXPECT_EQ(StringRef("DIFlagRValueReference"),
            DIDescriptor::getFlagString(DIDescriptor::FlagRValueReference));

  // FlagAccessibility actually equals FlagPublic.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DIDescriptor::getFlagString(DIDescriptor::FlagAccessibility));

  // Some other invalid flags.
  EXPECT_EQ(StringRef(), DIDescriptor::getFlagString(DIDescriptor::FlagPublic |
                                                     DIDescriptor::FlagVector));
  EXPECT_EQ(StringRef(),
            DIDescriptor::getFlagString(DIDescriptor::FlagFwdDecl |
                                        DIDescriptor::FlagArtificial));
  EXPECT_EQ(StringRef(), DIDescriptor::getFlagString(0xffff));
}

TEST(DIDescriptorTest, splitFlags) {
  // Some valid flags.
#define CHECK_SPLIT(FLAGS, VECTOR, REMAINDER)                                  \
  {                                                                            \
    SmallVector<unsigned, 8> V;                                                \
    EXPECT_EQ(REMAINDER, DIDescriptor::splitFlags(FLAGS, V));                  \
    EXPECT_TRUE(makeArrayRef(V).equals(VECTOR));                                \
  }
  CHECK_SPLIT(DIDescriptor::FlagPublic, {DIDescriptor::FlagPublic}, 0u);
  CHECK_SPLIT(DIDescriptor::FlagProtected, {DIDescriptor::FlagProtected}, 0u);
  CHECK_SPLIT(DIDescriptor::FlagPrivate, {DIDescriptor::FlagPrivate}, 0u);
  CHECK_SPLIT(DIDescriptor::FlagVector, {DIDescriptor::FlagVector}, 0u);
  CHECK_SPLIT(DIDescriptor::FlagRValueReference, {DIDescriptor::FlagRValueReference}, 0u);
  unsigned Flags[] = {DIDescriptor::FlagFwdDecl, DIDescriptor::FlagVector};
  CHECK_SPLIT(DIDescriptor::FlagFwdDecl | DIDescriptor::FlagVector, Flags, 0u);
  CHECK_SPLIT(0x100000u, {}, 0x100000u);
  CHECK_SPLIT(0x100000u | DIDescriptor::FlagVector, {DIDescriptor::FlagVector},
              0x100000u);
#undef CHECK_SPLIT
}

} // end namespace
