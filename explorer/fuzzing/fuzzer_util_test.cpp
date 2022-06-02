// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/fuzzer_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

static std::vector<llvm::StringRef>* carbon_files = nullptr;

// A workaround for https://github.com/carbon-language/carbon-lang/issues/1208.
TEST(FuzzerUtilTest, RunFuzzerOnCorpus) {
  int parsed_file_count = 0;
  for (const llvm::StringRef f : *carbon_files) {
    llvm::outs() << "Processing " << f << "\n";
    std::ifstream file(f.str(), std::ios::in);
    ASSERT_TRUE(file.is_open());
    std::stringstream contents;
    contents << file.rdbuf();
    // Parsing errors are ignored to make the fuzzer inputs less brittle as the
    // exlorer code changes. This also matches standard fuzzer behavior.
    if (auto carbon_proto = ParseCarbonTextProto(contents.str());
        carbon_proto.ok()) {
      ParseAndExecute(carbon_proto->compilation_unit());
      ++parsed_file_count;
    }
  }
  EXPECT_GT(parsed_file_count, 0);
}

TEST(FuzzerUtilTest, GetRunfilesFile) {
  EXPECT_THAT(*Internal::GetRunfilesFile("carbon/explorer/data/prelude.carbon"),
              testing::EndsWith("/prelude.carbon"));
  EXPECT_THAT(Internal::GetRunfilesFile("nonexistent-file").error().message(),
              testing::EndsWith("doesn't exist"));
}

TEST(FuzzerUtilTest, ParseCarbonTextProtoWithUnknownField) {
  const ErrorOr<Fuzzing::Carbon> carbon_proto =
      ParseCarbonTextProto(R"(
    compilation_unit {
      garbage: "value"
      declarations {
        choice {
          name: "Ch"
        }
      }
    })",
                           /*allow_unknown=*/true);
  ASSERT_TRUE(carbon_proto.ok());
  // No EqualsProto in gmock - https://github.com/google/googletest/issues/1761.
  EXPECT_EQ(carbon_proto->compilation_unit().declarations(0).choice().name(),
            "Ch");
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  Carbon::Testing::carbon_files =
      new std::vector<llvm::StringRef>(&argv[1], &argv[argc]);
  return RUN_ALL_TESTS();
}
