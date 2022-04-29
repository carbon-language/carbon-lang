// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/fuzzer_util.h"

#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>

#include <fstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

static std::vector<llvm::StringRef>* carbon_files = nullptr;

// A workaround for https://github.com/carbon-language/carbon-lang/issues/1208.
TEST(FuzzerUtilTest, RunFuzzerOnCorpus) {
  int file_count = 0;
  for (const llvm::StringRef f : *carbon_files) {
    llvm::outs() << "Processing " << f << "\n";
    std::ifstream file(f.str(), std::ios::in);
    ASSERT_TRUE(file.is_open());
    std::stringstream contents;
    contents << file.rdbuf();
    Fuzzing::Carbon carbon_proto;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(contents.str(),
                                                              &carbon_proto));
    ParseAndExecute(carbon_proto.compilation_unit());
    ++file_count;
  }
  EXPECT_GT(file_count, 0);
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  Carbon::Testing::carbon_files =
      new std::vector<llvm::StringRef>(&argv[1], &argv[argc]);
  return RUN_ALL_TESTS();
}
