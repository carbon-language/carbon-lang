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

#include "llvm/Support/MachO.h"

#include "lld/ReaderWriter/MachOLinkingContext.h"

using namespace llvm;
using namespace lld;

namespace {
class DarwinLdParserTest
    : public ParserTest<DarwinLdDriver, MachOLinkingContext> {
protected:
  const LinkingContext *linkingContext() override { return &_context; }
};
}

TEST_F(DarwinLdParserTest, Basic) {
  EXPECT_TRUE(parse("ld", "foo.o", "bar.o", nullptr));
  EXPECT_FALSE(_context.allowRemainingUndefines());
  EXPECT_FALSE(_context.deadStrip());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("foo.o", inputFile(0));
  EXPECT_EQ("bar.o", inputFile(1));
}

TEST_F(DarwinLdParserTest, Output) {
  EXPECT_TRUE(parse("ld", "-o", "my.out", "foo.o", nullptr));
  EXPECT_EQ("my.out", _context.outputPath());
}

TEST_F(DarwinLdParserTest, Dylib) {
  EXPECT_TRUE(parse("ld", "-dylib", "foo.o", nullptr));
  EXPECT_EQ(llvm::MachO::MH_DYLIB, _context.outputMachOType());
}

TEST_F(DarwinLdParserTest, Relocatable) {
  EXPECT_TRUE(parse("ld", "-r", "foo.o", nullptr));
  EXPECT_EQ(llvm::MachO::MH_OBJECT, _context.outputMachOType());
}

TEST_F(DarwinLdParserTest, Bundle) {
  EXPECT_TRUE(parse("ld", "-bundle", "foo.o", nullptr));
  EXPECT_EQ(llvm::MachO::MH_BUNDLE, _context.outputMachOType());
}

TEST_F(DarwinLdParserTest, Preload) {
  EXPECT_TRUE(parse("ld", "-preload", "foo.o", nullptr));
  EXPECT_EQ(llvm::MachO::MH_PRELOAD, _context.outputMachOType());
}

TEST_F(DarwinLdParserTest, Static) {
  EXPECT_TRUE(parse("ld", "-static", "foo.o", nullptr));
  EXPECT_EQ(llvm::MachO::MH_EXECUTE, _context.outputMachOType());
}

TEST_F(DarwinLdParserTest, Entry) {
  EXPECT_TRUE(parse("ld", "-e", "entryFunc", "foo.o", nullptr));
  EXPECT_EQ("entryFunc", _context.entrySymbolName());
}

TEST_F(DarwinLdParserTest, OutputPath) {
  EXPECT_TRUE(parse("ld", "-o", "foo", "foo.o", nullptr));
  EXPECT_EQ("foo", _context.outputPath());
}

TEST_F(DarwinLdParserTest, DeadStrip) {
  EXPECT_TRUE(parse("ld", "-arch", "x86_64", "-dead_strip", "foo.o", nullptr));
  EXPECT_TRUE(_context.deadStrip());
}

TEST_F(DarwinLdParserTest, DeadStripRootsExe) {
  EXPECT_TRUE(parse("ld", "-arch", "x86_64", "-dead_strip", "foo.o", nullptr));
  EXPECT_FALSE(_context.globalsAreDeadStripRoots());
}

TEST_F(DarwinLdParserTest, DeadStripRootsDylib) {
  EXPECT_TRUE(parse("ld", "-arch", "x86_64", "-dylib", "-dead_strip", "foo.o",
                    nullptr));
  EXPECT_TRUE(_context.globalsAreDeadStripRoots());
}

TEST_F(DarwinLdParserTest, Arch) {
  EXPECT_TRUE(parse("ld", "-arch", "x86_64", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::arch_x86_64, _context.arch());
  EXPECT_EQ((uint32_t)llvm::MachO::CPU_TYPE_X86_64, _context.getCPUType());
  EXPECT_EQ(llvm::MachO::CPU_SUBTYPE_X86_64_ALL, _context.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_x86) {
  EXPECT_TRUE(parse("ld", "-arch", "i386", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::arch_x86, _context.arch());
  EXPECT_EQ((uint32_t)llvm::MachO::CPU_TYPE_I386, _context.getCPUType());
  EXPECT_EQ(llvm::MachO::CPU_SUBTYPE_X86_ALL, _context.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_armv6) {
  EXPECT_TRUE(parse("ld", "-arch", "armv6", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::arch_armv6, _context.arch());
  EXPECT_EQ((uint32_t)llvm::MachO::CPU_TYPE_ARM, _context.getCPUType());
  EXPECT_EQ(llvm::MachO::CPU_SUBTYPE_ARM_V6, _context.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_armv7) {
  EXPECT_TRUE(parse("ld", "-arch", "armv7", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::arch_armv7, _context.arch());
  EXPECT_EQ((uint32_t)llvm::MachO::CPU_TYPE_ARM, _context.getCPUType());
  EXPECT_EQ(llvm::MachO::CPU_SUBTYPE_ARM_V7, _context.getCPUSubType());
}

TEST_F(DarwinLdParserTest, Arch_armv7s) {
  EXPECT_TRUE(parse("ld", "-arch", "armv7s", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::arch_armv7s, _context.arch());
  EXPECT_EQ((uint32_t)llvm::MachO::CPU_TYPE_ARM, _context.getCPUType());
  EXPECT_EQ(llvm::MachO::CPU_SUBTYPE_ARM_V7S, _context.getCPUSubType());
}

TEST_F(DarwinLdParserTest, MinMacOSX10_7) {
  EXPECT_TRUE(parse("ld", "-macosx_version_min", "10.7", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::OS::macOSX, _context.os());
  EXPECT_TRUE(_context.minOS("10.7", ""));
  EXPECT_FALSE(_context.minOS("10.8", ""));
}

TEST_F(DarwinLdParserTest, MinMacOSX10_8) {
  EXPECT_TRUE(parse("ld", "-macosx_version_min", "10.8.3", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::OS::macOSX, _context.os());
  EXPECT_TRUE(_context.minOS("10.7", ""));
  EXPECT_TRUE(_context.minOS("10.8", ""));
}

TEST_F(DarwinLdParserTest, iOS5) {
  EXPECT_TRUE(parse("ld", "-ios_version_min", "5.0", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::OS::iOS, _context.os());
  EXPECT_TRUE(_context.minOS("", "5.0"));
  EXPECT_FALSE(_context.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, iOS6) {
  EXPECT_TRUE(parse("ld", "-ios_version_min", "6.0", "foo.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::OS::iOS, _context.os());
  EXPECT_TRUE(_context.minOS("", "5.0"));
  EXPECT_TRUE(_context.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, iOS_Simulator5) {
  EXPECT_TRUE(parse("ld", "-ios_simulator_version_min", "5.0", "a.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::OS::iOS_simulator, _context.os());
  EXPECT_TRUE(_context.minOS("", "5.0"));
  EXPECT_FALSE(_context.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, iOS_Simulator6) {
  EXPECT_TRUE(parse("ld", "-ios_simulator_version_min", "6.0", "a.o", nullptr));
  EXPECT_EQ(MachOLinkingContext::OS::iOS_simulator, _context.os());
  EXPECT_TRUE(_context.minOS("", "5.0"));
  EXPECT_TRUE(_context.minOS("", "6.0"));
}

TEST_F(DarwinLdParserTest, compatibilityVersion) {
  EXPECT_TRUE(
      parse("ld", "-dylib", "-compatibility_version", "1.2.3", "a.o", nullptr));
  EXPECT_EQ(_context.compatibilityVersion(), 0x10203U);
}

TEST_F(DarwinLdParserTest, compatibilityVersionInvalidType) {
  EXPECT_FALSE(parse("ld", "-bundle", "-compatibility_version", "1.2.3", "a.o",
                     nullptr));
}

TEST_F(DarwinLdParserTest, compatibilityVersionInvalidValue) {
  EXPECT_FALSE(parse("ld", "-bundle", "-compatibility_version", "1,2,3", "a.o",
                     nullptr));
}

TEST_F(DarwinLdParserTest, currentVersion) {
  EXPECT_TRUE(
      parse("ld", "-dylib", "-current_version", "1.2.3", "a.o", nullptr));
  EXPECT_EQ(_context.currentVersion(), 0x10203U);
}

TEST_F(DarwinLdParserTest, currentVersionInvalidType) {
  EXPECT_FALSE(
      parse("ld", "-bundle", "-current_version", "1.2.3", "a.o", nullptr));
}

TEST_F(DarwinLdParserTest, currentVersionInvalidValue) {
  EXPECT_FALSE(
      parse("ld", "-bundle", "-current_version", "1,2,3", "a.o", nullptr));
}

TEST_F(DarwinLdParserTest, bundleLoader) {
  EXPECT_TRUE(
      parse("ld", "-bundle", "-bundle_loader", "/bin/ls", "a.o", nullptr));
  EXPECT_EQ(_context.bundleLoader(), "/bin/ls");
}

TEST_F(DarwinLdParserTest, bundleLoaderInvalidType) {
  EXPECT_FALSE(parse("ld", "-bundle_loader", "/bin/ls", "a.o", nullptr));
}

TEST_F(DarwinLdParserTest, deadStrippableDylib) {
  EXPECT_TRUE(
      parse("ld", "-dylib", "-mark_dead_strippable_dylib", "a.o", nullptr));
  EXPECT_EQ(true, _context.deadStrippableDylib());
}

TEST_F(DarwinLdParserTest, deadStrippableDylibInvalidType) {
  EXPECT_FALSE(parse("ld", "-mark_dead_strippable_dylib", "a.o", nullptr));
}

TEST_F(DarwinLdParserTest, llvmOptions) {
  EXPECT_TRUE(parse("ld", "-mllvm", "-debug-only", "-mllvm", "foo", "a.o", nullptr));
  const std::vector<const char *> &options = _context.llvmOptions();
  EXPECT_EQ(options.size(), 2UL);
  EXPECT_EQ(strcmp(options[0],"-debug-only"), 0);
  EXPECT_EQ(strcmp(options[1],"foo"), 0);
}
