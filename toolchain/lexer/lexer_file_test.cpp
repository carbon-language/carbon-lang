// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "re2/re2.h"
#include "toolchain/driver/driver_file_test_base.h"

namespace Carbon::Testing {
namespace {

class LexerFileTest : public DriverFileTestBase {
 public:
  explicit LexerFileTest(std::filesystem::path path)
      : DriverFileTestBase(std::move(path)),
        end_of_file_re_((R"((EndOfFile.*column: )( *\d+))")) {}

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"dump", "tokens", "%s"};
  }

  auto GetLineNumberReplacement(llvm::ArrayRef<llvm::StringRef> /*filenames*/)
      -> LineNumberReplacement override {
    return {.has_file = false,
            .pattern = R"(line: +(\d+))",
            // The `{{{{` becomes `{{`.
            .sub_for_formatv = "line: {{{{ *}}{0}"};
  }

  auto DoExtraCheckReplacements(std::string& check_line) -> void override {
    // Ignore the resulting column of EndOfFile because it's often the end of
    // the CHECK comment.
    RE2::Replace(&check_line, end_of_file_re_, R"(\1{{ *\\d+}})");
  }

 private:
  RE2 end_of_file_re_;
};

}  // namespace

CARBON_FILE_TEST_FACTORY(LexerFileTest);

}  // namespace Carbon::Testing
