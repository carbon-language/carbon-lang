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
  EXPECT_EQ("", T.getArchName());
  EXPECT_EQ("", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("-");
  EXPECT_EQ("", T.getArchName());
  EXPECT_EQ("", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("--");
  EXPECT_EQ("", T.getArchName());
  EXPECT_EQ("", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("---");
  EXPECT_EQ("", T.getArchName());
  EXPECT_EQ("", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("----");
  EXPECT_EQ("", T.getArchName());
  EXPECT_EQ("", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("-", T.getEnvironmentName());

  T = Triple("a");
  EXPECT_EQ("a", T.getArchName());
  EXPECT_EQ("", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("a-b");
  EXPECT_EQ("a", T.getArchName());
  EXPECT_EQ("b", T.getVendorName());
  EXPECT_EQ("", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("a-b-c");
  EXPECT_EQ("a", T.getArchName());
  EXPECT_EQ("b", T.getVendorName());
  EXPECT_EQ("c", T.getOSName());
  EXPECT_EQ("", T.getEnvironmentName());

  T = Triple("a-b-c-d");
  EXPECT_EQ("a", T.getArchName());
  EXPECT_EQ("b", T.getVendorName());
  EXPECT_EQ("c", T.getOSName());
  EXPECT_EQ("d", T.getEnvironmentName());
}

TEST(TripleTest, ParsedIDs) {
  Triple T;

  T = Triple("i386-apple-darwin");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::Apple, T.getVendor());
  EXPECT_EQ(Triple::Darwin, T.getOS());

  T = Triple("x86_64-pc-linux-gnu");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());

  T = Triple("powerpc-dunno-notsure");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T = Triple("huh");
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
}

TEST(TripleTest, MutateName) {
  Triple T;
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());

  T.setArchName("i386");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ("i386--", T.getTriple());

  T.setVendorName("pc");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ("i386-pc-", T.getTriple());

  T.setOSName("linux");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ("i386-pc-linux", T.getTriple());

  T.setEnvironmentName("gnu");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ("i386-pc-linux-gnu", T.getTriple());

  T.setOSName("freebsd");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::FreeBSD, T.getOS());
  EXPECT_EQ("i386-pc-freebsd-gnu", T.getTriple());

  T.setOSAndEnvironmentName("darwin");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Darwin, T.getOS());
  EXPECT_EQ("i386-pc-darwin", T.getTriple());
}

}
