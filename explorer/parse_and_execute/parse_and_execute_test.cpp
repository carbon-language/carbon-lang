// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

using ::testing::Eq;
using ::testing::RegisterTest;

static constexpr char PreludePath[] = "explorer/data/prelude.carbon";

TEST(ParseAndExecuteTest, Recursion) {
  std::string source = R"(
    package Test api;
    fn Main() -> i32 {
      return
  )";
  // A high depth that's expected to complete in a few seconds.
  static constexpr int Depth = 50000;
  for (int i = 0; i < Depth; ++i) {
    source += "if true then\n";
  }
  source += "1\n";
  for (int i = 0; i < Depth; ++i) {
    source += "else 0\n";
  }
  source += R"(
        ;
    }
  )";
  auto err = ParseAndExecute(PreludePath, source);
  ASSERT_FALSE(err.ok());
  EXPECT_THAT(err.error().message(),
              Eq("RUNTIME ERROR: overflow:1: stack overflow: too many "
                 "interpreter actions on stack"));
}

class ParseAndExecuteTestFile : public ::testing::Test {
 public:
  explicit ParseAndExecuteTestFile(llvm::StringRef path, bool parser_debug)
      : path_(path), parser_debug_(parser_debug) {}

  auto TestBody() -> void override {
    TraceStream trace_stream;
    std::string trace_stream_str;
    llvm::raw_string_ostream trace_stream_ostream(trace_stream_str);
    if (parser_debug_) {
      trace_stream.set_stream(&trace_stream_ostream);
    }

    auto result = ParseAndExecuteFile(PreludePath, path_.str(), parser_debug_,
                                      &trace_stream);
    if (filename().starts_with("fail_")) {
      EXPECT_FALSE(result.ok());
    } else {
      EXPECT_TRUE(result.ok()) << result.error();
    }
  }

  auto filename() -> llvm::StringRef {
    auto last_slash = path_.rfind("/");
    if (last_slash == llvm::StringRef::npos) {
      return path_;
    } else {
      return path_.substr(last_slash + 1);
    }
  }

 private:
  llvm::StringRef path_;
  bool parser_debug_;
};

static void RegisterTests(int argc, char** argv) {
  // Use RegisterTest instead of INSTANTIATE_TEST_CASE_P because of ordering
  // issues between container initialization and test instantiation by
  // InitGoogleTest.
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef path = argv[i];
    RegisterTest("ParseAndExecuteTestForFile", path.data(), nullptr,
                 path.data(), __FILE__, __LINE__, [=]() {
                   return new ParseAndExecuteTestFile(path,
                                                      /*parser_debug=*/false);
                 });
    // "limits" tests check for various limit conditions (such as an infinite
    // loop). The tests collectively don't test tracing because it creates
    // substantial additional overhead.
    if (!path.find("/limits/")) {
      RegisterTest("ParseAndExecuteTestForFile", path.data(), nullptr,
                   path.data(), __FILE__, __LINE__, [=]() {
                     return new ParseAndExecuteTestFile(path,
                                                        /*parser_debug=*/true);
                   });
    }
  }
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  Carbon::Testing::RegisterTests(argc, argv);
  return RUN_ALL_TESTS();
}
