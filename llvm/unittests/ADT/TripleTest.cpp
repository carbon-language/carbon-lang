//===----------- Triple.cpp - Triple unit tests ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

namespace {

TEST(TripleTest, BasicParsing) {
  Triple T;

  T = Triple("");
  EXPECT_EQ(T.getArchName(), "");
  EXPECT_EQ(T.getVendorName(), "");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("-");
  EXPECT_EQ(T.getArchName(), "");
  EXPECT_EQ(T.getVendorName(), "");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("--");
  EXPECT_EQ(T.getArchName(), "");
  EXPECT_EQ(T.getVendorName(), "");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("---");
  EXPECT_EQ(T.getArchName(), "");
  EXPECT_EQ(T.getVendorName(), "");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("----");
  EXPECT_EQ(T.getArchName(), "");
  EXPECT_EQ(T.getVendorName(), "");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "-");

  T = Triple("a");
  EXPECT_EQ(T.getArchName(), "a");
  EXPECT_EQ(T.getVendorName(), "");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("a-b");
  EXPECT_EQ(T.getArchName(), "a");
  EXPECT_EQ(T.getVendorName(), "b");
  EXPECT_EQ(T.getOSName(), "");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("a-b-c");
  EXPECT_EQ(T.getArchName(), "a");
  EXPECT_EQ(T.getVendorName(), "b");
  EXPECT_EQ(T.getOSName(), "c");
  EXPECT_EQ(T.getEnvironmentName(), "");

  T = Triple("a-b-c-d");
  EXPECT_EQ(T.getArchName(), "a");
  EXPECT_EQ(T.getVendorName(), "b");
  EXPECT_EQ(T.getOSName(), "c");
  EXPECT_EQ(T.getEnvironmentName(), "d");
}

TEST(TripleTest, ParsedIDs) {
  Triple T;

  T = Triple("i386-apple-darwin");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getVendor(), Triple::Apple);
  EXPECT_EQ(T.getOS(), Triple::Darwin);

  T = Triple("x86_64-pc-linux-gnu");
  EXPECT_EQ(T.getArch(), Triple::x86_64);
  EXPECT_EQ(T.getVendor(), Triple::PC);
  EXPECT_EQ(T.getOS(), Triple::Linux);

  T = Triple("powerpc-dunno-notsure");
  EXPECT_EQ(T.getArch(), Triple::ppc);
  EXPECT_EQ(T.getVendor(), Triple::UnknownVendor);
  EXPECT_EQ(T.getOS(), Triple::UnknownOS);

  T = Triple("huh");
  EXPECT_EQ(T.getArch(), Triple::UnknownArch);
}

TEST(TripleTest, MutateName) {
  Triple T;
  EXPECT_EQ(T.getArch(), Triple::UnknownArch);
  EXPECT_EQ(T.getVendor(), Triple::UnknownVendor);
  EXPECT_EQ(T.getOS(), Triple::UnknownOS);

  T.setArchName("i386");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getTriple(), "i386--");

  T.setVendorName("pc");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getVendor(), Triple::PC);
  EXPECT_EQ(T.getTriple(), "i386-pc-");

  T.setOSName("linux");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getVendor(), Triple::PC);
  EXPECT_EQ(T.getOS(), Triple::Linux);
  EXPECT_EQ(T.getTriple(), "i386-pc-linux");

  T.setEnvironmentName("gnu");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getVendor(), Triple::PC);
  EXPECT_EQ(T.getOS(), Triple::Linux);
  EXPECT_EQ(T.getTriple(), "i386-pc-linux-gnu");

  T.setOSName("freebsd");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getVendor(), Triple::PC);
  EXPECT_EQ(T.getOS(), Triple::FreeBSD);
  EXPECT_EQ(T.getTriple(), "i386-pc-freebsd-gnu");

  T.setOSAndEnvironmentName("darwin");
  EXPECT_EQ(T.getArch(), Triple::x86);
  EXPECT_EQ(T.getVendor(), Triple::PC);
  EXPECT_EQ(T.getOS(), Triple::Darwin);
  EXPECT_EQ(T.getTriple(), "i386-pc-darwin");
}

}
