//===- lld/unittest/DarwinLdDriverTest.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Darwin's ld driver tests.
///
//===----------------------------------------------------------------------===//

#include "DriverTest.h"

#include "lld/ReaderWriter/MachOTargetInfo.h"
#include "../../lib/ReaderWriter/MachO/MachOFormat.hpp"

using namespace llvm;
using namespace lld;

namespace {

class DarwinLdParserTest : public ParserTest<DarwinLdDriver, MachOTargetInfo> {
protected:
  virtual const TargetInfo *targetInfo() {
    return &_info;
  }
};

TEST_F(DarwinLdParserTest, Basic) {
  EXPECT_FALSE(parse("ld", "foo.o", "bar.o", nullptr));
  EXPECT_FALSE(_info.allowRemainingUndefines());
  EXPECT_FALSE(_info.deadStrip());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("foo.o", inputFile(0));
  EXPECT_EQ("bar.o", inputFile(1));
}

TEST_F(DarwinLdParserTest, Output) {
  parse("ld", "-o", "my.out", "foo.o", nullptr);
  EXPECT_EQ("my.out", _info.outputPath());
}

TEST_F(DarwinLdParserTest, Dylib) {
  EXPECT_FALSE(parse("ld", "-dylib", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_DYLIB, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Relocatable) {
  EXPECT_FALSE(parse("ld", "-r", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_OBJECT, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Bundle) {
  EXPECT_FALSE(parse("ld", "-bundle", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_BUNDLE, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Preload) {
  EXPECT_FALSE(parse("ld", "-preload", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_PRELOAD, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Static) {
  EXPECT_FALSE(parse("ld", "-static", "foo.o", nullptr));
  EXPECT_EQ(mach_o::MH_EXECUTE, _info.outputFileType());
}

TEST_F(DarwinLdParserTest, Entry) {
  EXPECT_FALSE(parse("ld", "-e", "entryFunc", "foo.o", nullptr));
  EXPECT_EQ("entryFunc", _info.entrySymbolName());
}

TEST_F(DarwinLdParserTest, OutputPath) {
  EXPECT_FALSE(parse("ld", "-o", "foo", "foo.o", nullptr));
  EXPECT_EQ("foo", _info.outputPath());
}

TEST_F(DarwinLdParserTest, DeadStrip) {
  EXPECT_FALSE(parse("ld", "-dead_strip", "foo.o", nullptr));
  EXPECT_TRUE(_info.deadStrip());
}

TEST_F(DarwinLdParserTest, DeadStripRootsExe) {
  parse("ld", "-dead_strip", "foo.o", nullptr);
  EXPECT_FALSE(_info.globalsAreDeadStripRoots());
}

TEST_F(DarwinLdParserTest, DeadStripRootsDylib) {
  parse("ld", "-dylib", "-dead_strip", "foo.o", nullptr);
  EXPECT_TRUE(_info.globalsAreDeadStripRoots());
}

TEST_F(DarwinLdParserTest, ForceLoadArchive) {
  parse("ld","-all_load", "foo.o", nullptr);
  EXPECT_TRUE(_info.forceLoadAllArchives());
}

TEST_F(DarwinLdParserTest, NoForceLoadArchive) {
  parse("ld", "foo.o", nullptr);
  EXPECT_FALSE(_info.forceLoadAllArchives());
}

TEST_F(DarwinLdParserTest, Arch) {
  EXPECT_FALSE(parse("ld", "-arch", "x86_64", "foo.o", nullptr));
  EXPECT_EQ(MachOTargetInfo::arch_x86_64, _info.arch());
  EXPECT_EQ(mach_o::CPU_TYPE_X86_64, _info.getCPUType());
  EXPECT_EQ(mach_o::CPU_SUBTYPE_X86_64_ALL, _info.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_x86) {
  parse("ld", "-arch", "i386", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::arch_x86, _info.arch());
  EXPECT_EQ(mach_o::CPU_TYPE_I386, _info.getCPUType());
  EXPECT_EQ(mach_o::CPU_SUBTYPE_X86_ALL, _info.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_armv6) {
  parse("ld", "-arch", "armv6", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::arch_armv6, _info.arch());
  EXPECT_EQ(mach_o::CPU_TYPE_ARM, _info.getCPUType());
  EXPECT_EQ(mach_o::CPU_SUBTYPE_ARM_V6, _info.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_armv7) {
  parse("ld", "-arch", "armv7", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::arch_armv7, _info.arch());
  EXPECT_EQ(mach_o::CPU_TYPE_ARM, _info.getCPUType());
  EXPECT_EQ(mach_o::CPU_SUBTYPE_ARM_V7, _info.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_armv7s) {
  parse("ld", "-arch", "armv7s", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::arch_armv7s, _info.arch());
  EXPECT_EQ(mach_o::CPU_TYPE_ARM, _info.getCPUType());
  EXPECT_EQ(mach_o::CPU_SUBTYPE_ARM_V7S, _info.getCPUSubType());
}

TEST_F(DarwinLdParserTest, MinMacOSX10_7) {
  parse("ld", "-macosx_version_min", "10.7", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::OS::macOSX, _info.os());
  EXPECT_TRUE(_info.minOS("10.7", ""));
  EXPECT_FALSE(_info.minOS("10.8", ""));
}

TEST_F(DarwinLdParserTest, MinMacOSX10_8) {
  parse("ld", "-macosx_version_min", "10.8.3", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::OS::macOSX, _info.os());
  EXPECT_TRUE(_info.minOS("10.7", ""));
  EXPECT_TRUE(_info.minOS("10.8", ""));
}

TEST_F(DarwinLdParserTest, iOS5) {
  parse("ld", "-ios_version_min", "5.0", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::OS::iOS, _info.os());
  EXPECT_TRUE(_info.minOS("", "5.0"));
  EXPECT_FALSE(_info.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, iOS6) {
  parse("ld", "-ios_version_min", "6.0", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::OS::iOS, _info.os());
  EXPECT_TRUE(_info.minOS("", "5.0"));
  EXPECT_TRUE(_info.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, iOS_Simulator5) {
  parse("ld", "-ios_simulator_version_min", "5.0", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::OS::iOS_simulator, _info.os());
  EXPECT_TRUE(_info.minOS("", "5.0"));
  EXPECT_FALSE(_info.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, iOS_Simulator6) {
  parse("ld", "-ios_simulator_version_min", "6.0", "foo.o", nullptr);
  EXPECT_EQ(MachOTargetInfo::OS::iOS_simulator, _info.os());
  EXPECT_TRUE(_info.minOS("", "5.0"));
  EXPECT_TRUE(_info.minOS("", "6.0"));
}



}  // end anonymous namespace
