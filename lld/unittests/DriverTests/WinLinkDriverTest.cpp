//===- lld/unittest/WinLinkDriverTest.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Windows link.exe driver tests.
///
//===----------------------------------------------------------------------===//

#include <stdarg.h>

#include "gtest/gtest.h"

#include "lld/Driver/Driver.h"
#include "lld/Driver/LinkerInput.h"
#include "lld/ReaderWriter/PECOFFTargetInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace lld;

namespace {

class ParserTest : public testing::Test {
protected:
  void SetUp() {
    os.reset(new raw_string_ostream(diags));
  }

  void parse(const char *args, ...) {
    std::vector<const char *> vec;
    vec.push_back("link.exe");
    vec.push_back(args);
    va_list ap;
    va_start(ap, args);
    while (const char *arg = va_arg(ap, const char *))
      vec.push_back(arg);
    va_end(ap);
    EXPECT_FALSE(WinLinkDriver::parse(vec.size(), &vec[0], info, *os));
  }

  PECOFFTargetInfo info;
  std::string diags;
  std::unique_ptr<raw_string_ostream> os;
};

TEST_F(ParserTest, Basic) {
  parse("-subsystem", "console", "-out", "a.exe", "a.obj", "b.obj", "c.obj",
        nullptr);

  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, info.getSubsystem());
  EXPECT_EQ("a.exe", info.outputPath());

  const std::vector<LinkerInput> &inputFiles = info.inputFiles();
  EXPECT_EQ((size_t)3, inputFiles.size());
  EXPECT_EQ("a.obj", inputFiles[0].getPath());
  EXPECT_EQ("b.obj", inputFiles[1].getPath());
  EXPECT_EQ("c.obj", inputFiles[2].getPath());
}

TEST_F(ParserTest, WindowsStyleOption) {
  parse("/subsystem:console", "/out:a.exe", "a.obj", nullptr);

  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, info.getSubsystem());
  EXPECT_EQ("a.exe", info.outputPath());

  const std::vector<LinkerInput> &inputFiles = info.inputFiles();
  EXPECT_EQ((size_t)1, inputFiles.size());
  EXPECT_EQ("a.obj", inputFiles[0].getPath());
}

TEST_F(ParserTest, NoFileExtension) {
  parse("foo", "bar", nullptr);

  EXPECT_EQ("foo.exe", info.outputPath());

  const std::vector<LinkerInput> &inputFiles = info.inputFiles();
  EXPECT_EQ((size_t)2, inputFiles.size());
  EXPECT_EQ("foo.obj", inputFiles[0].getPath());
  EXPECT_EQ("bar.obj", inputFiles[1].getPath());
}

TEST_F(ParserTest, NonStandardFileExtension) {
  parse("foo.o", nullptr);

  EXPECT_EQ("foo.exe", info.outputPath());

  const std::vector<LinkerInput> &inputFiles = info.inputFiles();
  EXPECT_EQ((size_t)1, inputFiles.size());
  EXPECT_EQ("foo.o", inputFiles[0].getPath());
}

}  // end anonymous namespace
