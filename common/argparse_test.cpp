// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/argparse.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/test_raw_ostream.h"

namespace Carbon::Testing {
namespace {

TEST(ArgParserTest, Basic) {
  ArgParser::Flag flag1 = {
    .name = "flag1",
  };

  ArgParser::Command sub1 = {
    .name = "sub1",
  };

  ArgParser::Command sub2 = {
    .name = "sub2",
    .flags = {
      flag1,
    },
  };

  auto parser = ArgParser::Make({
    .command = {
      .name = "command",
    },
    .subcommands = {
      sub1,
      sub2,
    },
  });
  (void)parser;
}

}  // namespace
}  // namespace Carbon::Testing
