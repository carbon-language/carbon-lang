//===- lld/unittest/DriverTest.h ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdarg.h>

#include "gtest/gtest.h"

#include "lld/Driver/Driver.h"
#include "lld/Driver/LinkerInput.h"

#include "llvm/Support/raw_ostream.h"

namespace {

using namespace llvm;
using namespace lld;

template<typename Driver, typename TargetInfo>
class ParserTest : public testing::Test {
protected:
  void SetUp() {
    os.reset(new raw_string_ostream(diags));
  }

  virtual TargetInfo *doParse(int argc, const char **argv,
                              raw_ostream &diag) = 0;

  void parse(const char *args, ...) {
    // Construct command line options from varargs.
    std::vector<const char *> vec;
    vec.push_back(args);
    va_list ap;
    va_start(ap, args);
    while (const char *arg = va_arg(ap, const char *))
      vec.push_back(arg);
    va_end(ap);

    // Call the parser.
    info.reset(doParse(vec.size(), &vec[0], *os));

    // Copy the output file name for the sake of convenience.
    if (info)
      for (const LinkerInput &input : info->inputFiles())
        inputFiles.push_back(input.getPath().str());
  }

  std::unique_ptr<TargetInfo> info;
  std::string diags;
  std::unique_ptr<raw_string_ostream> os;
  std::vector<std::string> inputFiles;
};

}
