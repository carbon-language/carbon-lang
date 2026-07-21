// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/mismatched_brackets.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Lex {
namespace {

using ::testing::SizeIs;

class MismatchedBracketsTest : public ::testing::Test {
 protected:
  auto MakeToken(int32_t index, BracketTokenKind kind, int32_t line,
                 int32_t indent, bool is_at_end_of_line = false,
                 bool is_struct_brace = false) -> MismatchedBracketToken {
    return MismatchedBracketToken{
        .token_index = TokenIndex(index),
        .kind = kind,
        .line = line,
        .line_indent = indent,
        .column = indent + 1,
        .is_at_end_of_line = is_at_end_of_line,
        .is_struct_brace = is_struct_brace,
    };
  }
};

TEST_F(MismatchedBracketsTest, HandlesEmptyTokens) {
  llvm::SmallVector<MismatchedBracketToken> tokens;
  auto corrections = FixMismatchedBrackets(tokens);
  EXPECT_TRUE(corrections.empty());
}

TEST_F(MismatchedBracketsTest, HandlesWellBalancedCode) {
  // 1  fn F() {
  // 2    if (x) {
  // 3      y;
  // 4    }
  // 5  }
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::StatementIntroducer, 1, 1),
      MakeToken(1, BracketTokenKind::Other, 1, 1),
      MakeToken(2, BracketTokenKind::OpenParen, 1, 1),
      MakeToken(3, BracketTokenKind::CloseParen, 1, 1),
      MakeToken(4, BracketTokenKind::OpenCurlyBrace, 1, 1,
                /*is_at_end_of_line=*/true),
      MakeToken(5, BracketTokenKind::StatementIntroducer, 2, 3),
      MakeToken(6, BracketTokenKind::OpenParen, 2, 3),
      MakeToken(7, BracketTokenKind::Other, 2, 3),
      MakeToken(8, BracketTokenKind::CloseParen, 2, 3),
      MakeToken(9, BracketTokenKind::OpenCurlyBrace, 2, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(10, BracketTokenKind::Other, 3, 5),
      MakeToken(11, BracketTokenKind::Semi, 3, 5, /*is_at_end_of_line=*/true),
      MakeToken(12, BracketTokenKind::CloseCurlyBrace, 4, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(13, BracketTokenKind::CloseCurlyBrace, 5, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  EXPECT_TRUE(corrections.empty());
}

TEST_F(MismatchedBracketsTest, FixesMissingOpenBraceAfterIf) {
  // 1  fn F() {
  // 2    if (thing1)
  // 3      thing2;
  // 4    }
  // 5  }
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::StatementIntroducer, 1, 1),
      MakeToken(1, BracketTokenKind::Other, 1, 1),
      MakeToken(2, BracketTokenKind::OpenParen, 1, 1),
      MakeToken(3, BracketTokenKind::CloseParen, 1, 1),
      MakeToken(4, BracketTokenKind::OpenCurlyBrace, 1, 1,
                /*is_at_end_of_line=*/true),
      MakeToken(5, BracketTokenKind::StatementIntroducer, 2, 3),
      MakeToken(6, BracketTokenKind::OpenParen, 2, 3),
      MakeToken(7, BracketTokenKind::Other, 2, 3),
      MakeToken(8, BracketTokenKind::CloseParen, 2, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(9, BracketTokenKind::Other, 3, 5),
      MakeToken(10, BracketTokenKind::Semi, 3, 5, /*is_at_end_of_line=*/true),
      MakeToken(11, BracketTokenKind::CloseCurlyBrace, 4, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(12, BracketTokenKind::CloseCurlyBrace, 5, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  EXPECT_FALSE(corrections.empty());

  bool inserted_open_brace = false;
  for (const auto& corr : corrections) {
    if (corr.fix_action == BracketFixAction::InsertBefore &&
        corr.fix_token_kind == TokenKind::OpenCurlyBrace) {
      inserted_open_brace = true;
      // Should be inserted before line 3 tokens (e.g. token 9).
      EXPECT_EQ(corr.fix_token_index.index, 9);
      EXPECT_EQ(corr.diagnostic_kind, BracketDiagnosticKind::UnmatchedClosing);
    }
  }
  EXPECT_TRUE(inserted_open_brace);
}

TEST_F(MismatchedBracketsTest, HandlesMultiLineDeclarationHeader) {
  // 1  fn F[T: type]
  // 2      (x: T) {
  // 3    foo();
  // 4  }
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::StatementIntroducer, 1, 1),
      MakeToken(1, BracketTokenKind::Other, 1, 1),
      MakeToken(2, BracketTokenKind::OpenSquareBracket, 1, 1),
      MakeToken(3, BracketTokenKind::Other, 1, 1),
      MakeToken(4, BracketTokenKind::CloseSquareBracket, 1, 1,
                /*is_at_end_of_line=*/true),
      MakeToken(5, BracketTokenKind::OpenParen, 2, 5),
      MakeToken(6, BracketTokenKind::Other, 2, 5),
      MakeToken(7, BracketTokenKind::CloseParen, 2, 5),
      MakeToken(8, BracketTokenKind::OpenCurlyBrace, 2, 5,
                /*is_at_end_of_line=*/true),
      MakeToken(9, BracketTokenKind::Other, 3, 3),
      MakeToken(10, BracketTokenKind::OpenParen, 3, 3),
      MakeToken(11, BracketTokenKind::CloseParen, 3, 3),
      MakeToken(12, BracketTokenKind::Semi, 3, 3, /*is_at_end_of_line=*/true),
      MakeToken(13, BracketTokenKind::CloseCurlyBrace, 4, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  EXPECT_TRUE(corrections.empty());
}

TEST_F(MismatchedBracketsTest, FixesOmittedOpenBraceWithMultiLineHeader) {
  // 1  fn F[T: type]
  // 2      (x: T)
  // 3    foo();
  // 4    bar();
  // 5  }
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::StatementIntroducer, 1, 1),
      MakeToken(1, BracketTokenKind::Other, 1, 1),
      MakeToken(2, BracketTokenKind::OpenSquareBracket, 1, 1),
      MakeToken(3, BracketTokenKind::Other, 1, 1),
      MakeToken(4, BracketTokenKind::CloseSquareBracket, 1, 1,
                /*is_at_end_of_line=*/true),
      MakeToken(5, BracketTokenKind::OpenParen, 2, 5),
      MakeToken(6, BracketTokenKind::Other, 2, 5),
      MakeToken(7, BracketTokenKind::CloseParen, 2, 5,
                /*is_at_end_of_line=*/true),
      MakeToken(8, BracketTokenKind::Other, 3, 3),
      MakeToken(9, BracketTokenKind::Semi, 3, 3, /*is_at_end_of_line=*/true),
      MakeToken(10, BracketTokenKind::Other, 4, 3),
      MakeToken(11, BracketTokenKind::Semi, 4, 3, /*is_at_end_of_line=*/true),
      MakeToken(12, BracketTokenKind::CloseCurlyBrace, 5, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  EXPECT_FALSE(corrections.empty());

  bool inserted_open_brace = false;
  for (const auto& corr : corrections) {
    if (corr.fix_action == BracketFixAction::InsertBefore &&
        corr.fix_token_kind == TokenKind::OpenCurlyBrace) {
      inserted_open_brace = true;
      // Should be inserted before line 3 statement (token 8).
      EXPECT_EQ(corr.fix_token_index.index, 8);
    }
  }
  EXPECT_TRUE(inserted_open_brace);
}

TEST_F(MismatchedBracketsTest, HandlesUnmatchedClosingBrace) {
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::CloseCurlyBrace, 1, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  ASSERT_THAT(corrections, SizeIs(1));
  EXPECT_EQ(corrections[0].diagnostic_kind,
            BracketDiagnosticKind::UnmatchedClosing);
  EXPECT_EQ(corrections[0].diagnostic_token_index.index, 0);
  EXPECT_EQ(corrections[0].fix_action, BracketFixAction::ReplaceWithError);
  EXPECT_EQ(corrections[0].fix_token_index.index, 0);
}

TEST_F(MismatchedBracketsTest, HandlesUnclosedOpeningBraceAtEOF) {
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::OpenCurlyBrace, 1, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  ASSERT_THAT(corrections, SizeIs(1));
  EXPECT_EQ(corrections[0].diagnostic_kind,
            BracketDiagnosticKind::UnmatchedOpening);
  EXPECT_EQ(corrections[0].diagnostic_token_index.index, 0);
}

TEST_F(MismatchedBracketsTest, MissingClosingBraceBeforeSiblingFunction) {
  // 1 class Grid {
  // 2   fn Check4() {
  // 3     return;
  //       // missing }
  // 4   fn Check3() {
  // 5     return;
  // 6   }
  // 7 }
  llvm::SmallVector<MismatchedBracketToken> tokens = {
      MakeToken(0, BracketTokenKind::StatementIntroducer, 1, 1),
      MakeToken(1, BracketTokenKind::OpenCurlyBrace, 1, 1,
                /*is_at_end_of_line=*/true),
      MakeToken(2, BracketTokenKind::StatementIntroducer, 2, 3),
      MakeToken(3, BracketTokenKind::OpenParen, 2, 3),
      MakeToken(4, BracketTokenKind::CloseParen, 2, 3),
      MakeToken(5, BracketTokenKind::OpenCurlyBrace, 2, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(6, BracketTokenKind::StatementIntroducer, 3, 5),
      MakeToken(7, BracketTokenKind::Semi, 3, 5, /*is_at_end_of_line=*/true),
      MakeToken(8, BracketTokenKind::StatementIntroducer, 4, 3),
      MakeToken(9, BracketTokenKind::OpenParen, 4, 3),
      MakeToken(10, BracketTokenKind::CloseParen, 4, 3),
      MakeToken(11, BracketTokenKind::OpenCurlyBrace, 4, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(12, BracketTokenKind::StatementIntroducer, 5, 5),
      MakeToken(13, BracketTokenKind::Semi, 5, 5, /*is_at_end_of_line=*/true),
      MakeToken(14, BracketTokenKind::CloseCurlyBrace, 6, 3,
                /*is_at_end_of_line=*/true),
      MakeToken(15, BracketTokenKind::CloseCurlyBrace, 7, 1,
                /*is_at_end_of_line=*/true),
  };

  auto corrections = FixMismatchedBrackets(tokens);
  ASSERT_FALSE(corrections.empty());
  EXPECT_EQ(corrections[0].fix_action, BracketFixAction::InsertBefore);
  EXPECT_EQ(corrections[0].fix_token_index.index,
            8);  // Should insert before token 8 (fn Check3).
}

TEST_F(MismatchedBracketsTest, PathologicalInputFallsBackSafely) {
  llvm::SmallVector<MismatchedBracketToken> tokens;
  for (int32_t i = 0; i < 200; ++i) {
    tokens.push_back(MakeToken(i,
                               (i % 2 == 0) ? BracketTokenKind::OpenParen
                                            : BracketTokenKind::CloseCurlyBrace,
                               i + 1, (i % 4) * 2));
  }
}

}  // namespace
}  // namespace Carbon::Lex
