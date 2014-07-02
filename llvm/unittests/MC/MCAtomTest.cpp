//===- llvm/unittest/MC/MCAtomTest.cpp - Instructions unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAnalysis/MCAtom.h"
#include "llvm/MC/MCAnalysis/MCModule.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(MCAtomTest, MCDataSize) {
  MCModule M;
  MCDataAtom *Atom = M.createDataAtom(0, 0);
  EXPECT_EQ(uint64_t(0), Atom->getEndAddr());
  Atom->addData(0);
  EXPECT_EQ(uint64_t(0), Atom->getEndAddr());
  Atom->addData(1);
  EXPECT_EQ(uint64_t(1), Atom->getEndAddr());
  Atom->addData(2);
  EXPECT_EQ(uint64_t(2), Atom->getEndAddr());
  EXPECT_EQ(size_t(3), Atom->getData().size());
}

}  // end anonymous namespace
}  // end namespace llvm
