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
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("-");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("--");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("---");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("----");
  EXPECT_EQ("", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("-", T.getEnvironmentName().str());

  T = Triple("a");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("a-b");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("b", T.getVendorName().str());
  EXPECT_EQ("", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("a-b-c");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("b", T.getVendorName().str());
  EXPECT_EQ("c", T.getOSName().str());
  EXPECT_EQ("", T.getEnvironmentName().str());

  T = Triple("a-b-c-d");
  EXPECT_EQ("a", T.getArchName().str());
  EXPECT_EQ("b", T.getVendorName().str());
  EXPECT_EQ("c", T.getOSName().str());
  EXPECT_EQ("d", T.getEnvironmentName().str());
}

TEST(TripleTest, ParsedIDs) {
  Triple T;

  T = Triple("i386-apple-darwin");
  EXPECT_EQ(Triple::x86, T.getArch());
  EXPECT_EQ(Triple::Apple, T.getVendor());
  EXPECT_EQ(Triple::Darwin, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("x86_64-pc-linux-gnu");
  EXPECT_EQ(Triple::x86_64, T.getArch());
  EXPECT_EQ(Triple::PC, T.getVendor());
  EXPECT_EQ(Triple::Linux, T.getOS());
  EXPECT_EQ(Triple::GNU, T.getEnvironment());

  T = Triple("powerpc-dunno-notsure");
  EXPECT_EQ(Triple::ppc, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

  T = Triple("arm-none-eabi");
  EXPECT_EQ(Triple::arm, T.getArch());
  EXPECT_EQ(Triple::NoVendor, T.getVendor());
  EXPECT_EQ(Triple::NoOS, T.getOS());
  EXPECT_EQ(Triple::EABI, T.getEnvironment());

  T = Triple("huh");
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
}

static std::string Join(StringRef A, StringRef B, StringRef C) {
  std::string Str = A; Str += '-'; Str += B; Str += '-'; Str += C;
  return Str;
}

static std::string Join(StringRef A, StringRef B, StringRef C, StringRef D) {
  std::string Str = A; Str += '-'; Str += B; Str += '-'; Str += C; Str += '-';
  Str += D; return Str;
}

TEST(TripleTest, Normalization) {

  EXPECT_EQ("", Triple::normalize(""));
  EXPECT_EQ("-", Triple::normalize("-"));
  EXPECT_EQ("--", Triple::normalize("--"));
  EXPECT_EQ("---", Triple::normalize("---"));
  EXPECT_EQ("----", Triple::normalize("----"));

  EXPECT_EQ("a", Triple::normalize("a"));
  EXPECT_EQ("a-b", Triple::normalize("a-b"));
  EXPECT_EQ("a-b-c", Triple::normalize("a-b-c"));
  EXPECT_EQ("a-b-c-d", Triple::normalize("a-b-c-d"));

  EXPECT_EQ("i386-b-c", Triple::normalize("i386-b-c"));
  EXPECT_EQ("i386-a-c", Triple::normalize("a-i386-c"));
  EXPECT_EQ("i386-a-b", Triple::normalize("a-b-i386"));
  EXPECT_EQ("i386-a-b-c", Triple::normalize("a-b-c-i386"));

  EXPECT_EQ("a-pc-c", Triple::normalize("a-pc-c"));
  EXPECT_EQ("-pc-b-c", Triple::normalize("pc-b-c"));
  EXPECT_EQ("a-pc-b", Triple::normalize("a-b-pc"));
  EXPECT_EQ("a-pc-b-c", Triple::normalize("a-b-c-pc"));

  EXPECT_EQ("a-b-linux", Triple::normalize("a-b-linux"));
  EXPECT_EQ("--linux-b-c", Triple::normalize("linux-b-c"));
  EXPECT_EQ("a--linux-c", Triple::normalize("a-linux-c"));

  EXPECT_EQ("i386-pc-a", Triple::normalize("a-pc-i386"));
  EXPECT_EQ("i386-pc-", Triple::normalize("-pc-i386"));
  EXPECT_EQ("-pc-linux-c", Triple::normalize("linux-pc-c"));
  EXPECT_EQ("-pc-linux", Triple::normalize("linux-pc-"));

  EXPECT_EQ("i386", Triple::normalize("i386"));
  EXPECT_EQ("-pc", Triple::normalize("pc"));
  EXPECT_EQ("--linux", Triple::normalize("linux"));

  EXPECT_EQ("x86_64--linux-gnu", Triple::normalize("x86_64-gnu-linux"));

  // Check that normalizing a permutated set of valid components returns a
  // triple with the unpermuted components.
  StringRef C[4];
  C[3] = "environment";
  for (int Arch = 1+Triple::UnknownArch; Arch < Triple::InvalidArch; ++Arch) {
    C[0] = Triple::getArchTypeName(Triple::ArchType(Arch));
    for (int Vendor = 1+Triple::UnknownVendor; Vendor <= Triple::PC;
         ++Vendor) {
      C[1] = Triple::getVendorTypeName(Triple::VendorType(Vendor));
      for (int OS = 1+Triple::UnknownOS; OS <= Triple::Minix; ++OS) {
        C[2] = Triple::getOSTypeName(Triple::OSType(OS));

        std::string E = Join(C[0], C[1], C[2]);
        std::string F = Join(C[0], C[1], C[2], C[3]);
        EXPECT_EQ(E, Triple::normalize(Join(C[0], C[1], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[0], C[1], C[2], C[3])));

        // If a value has multiple interpretations, then the permutation
        // test will inevitably fail.  Currently this is only the case for
        // "psp" which parses as both an architecture and an O/S.
        if (OS == Triple::Psp)
          continue;

        EXPECT_EQ(E, Triple::normalize(Join(C[0], C[2], C[1])));
        EXPECT_EQ(E, Triple::normalize(Join(C[1], C[2], C[0])));
        EXPECT_EQ(E, Triple::normalize(Join(C[1], C[0], C[2])));
        EXPECT_EQ(E, Triple::normalize(Join(C[2], C[0], C[1])));
        EXPECT_EQ(E, Triple::normalize(Join(C[2], C[1], C[0])));

        EXPECT_EQ(F, Triple::normalize(Join(C[0], C[1], C[3], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[0], C[2], C[3], C[1])));
        EXPECT_EQ(F, Triple::normalize(Join(C[0], C[2], C[1], C[3])));
        EXPECT_EQ(F, Triple::normalize(Join(C[0], C[3], C[1], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[0], C[3], C[2], C[1])));
        EXPECT_EQ(F, Triple::normalize(Join(C[1], C[2], C[3], C[0])));
        EXPECT_EQ(F, Triple::normalize(Join(C[1], C[2], C[0], C[3])));
        EXPECT_EQ(F, Triple::normalize(Join(C[1], C[3], C[0], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[1], C[3], C[2], C[0])));
        EXPECT_EQ(F, Triple::normalize(Join(C[1], C[0], C[2], C[3])));
        EXPECT_EQ(F, Triple::normalize(Join(C[1], C[0], C[3], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[2], C[3], C[0], C[1])));
        EXPECT_EQ(F, Triple::normalize(Join(C[2], C[3], C[1], C[0])));
        EXPECT_EQ(F, Triple::normalize(Join(C[2], C[0], C[1], C[3])));
        EXPECT_EQ(F, Triple::normalize(Join(C[2], C[0], C[3], C[1])));
        EXPECT_EQ(F, Triple::normalize(Join(C[2], C[1], C[3], C[0])));
        EXPECT_EQ(F, Triple::normalize(Join(C[2], C[1], C[0], C[3])));
        EXPECT_EQ(F, Triple::normalize(Join(C[3], C[0], C[1], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[3], C[0], C[2], C[1])));
        EXPECT_EQ(F, Triple::normalize(Join(C[3], C[1], C[2], C[0])));
        EXPECT_EQ(F, Triple::normalize(Join(C[3], C[1], C[0], C[2])));
        EXPECT_EQ(F, Triple::normalize(Join(C[3], C[2], C[0], C[1])));
        EXPECT_EQ(F, Triple::normalize(Join(C[3], C[2], C[1], C[0])));
      }
    }
  }

  EXPECT_EQ("a-b-psp", Triple::normalize("a-b-psp"));
  EXPECT_EQ("psp-b-c", Triple::normalize("psp-b-c"));

  // Various real-world funky triples.  The value returned by GCC's config.sub
  // is given in the comment.
  EXPECT_EQ("i386--mingw32", Triple::normalize("i386-mingw32")); // i386-pc-mingw32
  EXPECT_EQ("x86_64--linux-gnu", Triple::normalize("x86_64-linux-gnu")); // x86_64-pc-linux-gnu
  EXPECT_EQ("i486--linux-gnu", Triple::normalize("i486-linux-gnu")); // i486-pc-linux-gnu
  EXPECT_EQ("i386-redhat-linux", Triple::normalize("i386-redhat-linux")); // i386-redhat-linux-gnu
  EXPECT_EQ("i686--linux", Triple::normalize("i686-linux")); // i686-pc-linux-gnu
}

TEST(TripleTest, MutateName) {
  Triple T;
  EXPECT_EQ(Triple::UnknownArch, T.getArch());
  EXPECT_EQ(Triple::UnknownVendor, T.getVendor());
  EXPECT_EQ(Triple::UnknownOS, T.getOS());
  EXPECT_EQ(Triple::UnknownEnvironment, T.getEnvironment());

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
