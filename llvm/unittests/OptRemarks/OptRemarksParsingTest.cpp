//===- unittest/Support/OptRemarksParsingTest.cpp - OptTable tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/OptRemarks.h"
#include "gtest/gtest.h"

using namespace llvm;

template <size_t N> bool tryParse(const char (&Buf)[N]) {
  LLVMOptRemarkParserRef Parser = LLVMOptRemarkParserCreate(Buf, N - 1);
  LLVMOptRemarkEntry *Remark = nullptr;
  while (LLVMOptRemarkEntry *NewRemark = LLVMOptRemarkParserGetNext(Parser)) {
    EXPECT_TRUE(Remark == nullptr); // Only one remark per test.
    Remark = NewRemark;
  }
  EXPECT_TRUE(Remark != nullptr); // We need *exactly* one remark per test.
  bool HasError = LLVMOptRemarkParserHasError(Parser);
  LLVMOptRemarkParserDispose(Parser);
  return !HasError;
}

template <size_t N>
bool parseExpectError(const char (&Buf)[N], const char *Error) {
  LLVMOptRemarkParserRef Parser = LLVMOptRemarkParserCreate(Buf, N - 1);
  LLVMOptRemarkEntry *Remark = nullptr;
  while (LLVMOptRemarkEntry *NewRemark = LLVMOptRemarkParserGetNext(Parser)) {
    EXPECT_FALSE(NewRemark);
  }
  EXPECT_TRUE(Remark == nullptr); // We are parsing only one malformed remark.
  EXPECT_TRUE(LLVMOptRemarkParserHasError(Parser));
  bool MatchesError =
      StringRef(LLVMOptRemarkParserGetErrorMessage(Parser)).contains(Error);
  LLVMOptRemarkParserDispose(Parser);

  return MatchesError;
}

TEST(OptRemarks, OptRemarksParsingEmpty) {
  StringRef Buf = R"YAML(
)YAML";
  LLVMOptRemarkParserRef Parser =
      LLVMOptRemarkParserCreate(Buf.data(), Buf.size());
  LLVMOptRemarkEntry *NewRemark = LLVMOptRemarkParserGetNext(Parser);
  EXPECT_TRUE(NewRemark == nullptr); // No remark expected.
  EXPECT_TRUE(LLVMOptRemarkParserHasError(Parser));
  EXPECT_TRUE(StringRef(LLVMOptRemarkParserGetErrorMessage(Parser))
                  .contains("document root is not of mapping type."));
  LLVMOptRemarkParserDispose(Parser);
}

TEST(OptRemarks, OptRemarksParsingGood) {
  EXPECT_TRUE(tryParse(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
DebugLoc: { File: file.c, Line: 3, Column: 12 }
Function: foo
Args:
  - Callee: bar
  - String: ' will not be inlined into '
  - Caller: foo
    DebugLoc: { File: file.c, Line: 2, Column: 0 }
  - String: ' because its definition is unavailable'
)YAML"));

  // No debug loc should also pass.
  EXPECT_TRUE(tryParse(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
Args:
  - Callee: bar
  - String: ' will not be inlined into '
  - Caller: foo
    DebugLoc: { File: file.c, Line: 2, Column: 0 }
  - String: ' because its definition is unavailable'
)YAML"));

  // No args is also ok.
  EXPECT_TRUE(tryParse(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
DebugLoc: { File: file.c, Line: 3, Column: 12 }
Function: foo
)YAML"));

  // Different order.
  EXPECT_TRUE(tryParse(R"YAML(
--- !Missed
DebugLoc: { Line: 3, Column: 12, File: file.c }
Function: foo
Name: NoDefinition
Args:
  - Callee: bar
  - String: ' will not be inlined into '
  - Caller: foo
    DebugLoc: { File: file.c, Line: 2, Column: 0 }
  - String: ' because its definition is unavailable'
Pass: inline
)YAML"));
}

// Mandatory common part of a remark.
#define COMMON_REMARK "\nPass: inline\nName: NoDefinition\nFunction: foo\n"
// Test all the types.
TEST(OptRemarks, OptRemarksParsingTypes) {
  // Type: Passed
  EXPECT_TRUE(tryParse("--- !Passed" COMMON_REMARK));
  // Type: Missed
  EXPECT_TRUE(tryParse("--- !Missed" COMMON_REMARK));
  // Type: Analysis
  EXPECT_TRUE(tryParse("--- !Analysis" COMMON_REMARK));
  // Type: AnalysisFPCompute
  EXPECT_TRUE(tryParse("--- !AnalysisFPCompute" COMMON_REMARK));
  // Type: AnalysisAliasing
  EXPECT_TRUE(tryParse("--- !AnalysisAliasing" COMMON_REMARK));
  // Type: Failure
  EXPECT_TRUE(tryParse("--- !Failure" COMMON_REMARK));
}
#undef COMMON_REMARK

TEST(OptRemarks, OptRemarksParsingMissingFields) {
  // No type.
  EXPECT_TRUE(parseExpectError(R"YAML(
---
Pass: inline
Name: NoDefinition
Function: foo
)YAML",
                               "error: Type, Pass, Name or Function missing."));
  // No pass.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Name: NoDefinition
Function: foo
)YAML",
                               "error: Type, Pass, Name or Function missing."));
  // No name.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Function: foo
)YAML",
                               "error: Type, Pass, Name or Function missing."));
  // No function.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
)YAML",
                               "error: Type, Pass, Name or Function missing."));
  // Debug loc but no file.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { Line: 3, Column: 12 }
)YAML",
                               "DebugLoc node incomplete."));
  // Debug loc but no line.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Column: 12 }
)YAML",
                               "DebugLoc node incomplete."));
  // Debug loc but no column.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Line: 3 }
)YAML",
                               "DebugLoc node incomplete."));
}

TEST(OptRemarks, OptRemarksParsingWrongTypes) {
  // Wrong debug loc type.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: foo
)YAML",
                               "expected a value of mapping type."));
  // Wrong line type.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Line: b, Column: 12 }
)YAML",
                               "expected a value of integer type."));
  // Wrong column type.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Line: 3, Column: c }
)YAML",
                               "expected a value of integer type."));
  // Wrong args type.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
Args: foo
)YAML",
                               "wrong value type for key."));
  // Wrong key type.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
{ A: a }: inline
Name: NoDefinition
Function: foo
)YAML",
                               "key is not a string."));
  // Debug loc with unknown entry.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Column: 12, Unknown: 12 }
)YAML",
                               "unknown entry in DebugLoc map."));
  // Unknown entry.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Unknown: inline
)YAML",
                               "unknown key."));
  // Not a scalar.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: { File: a, Line: 1, Column: 2 }
Name: NoDefinition
Function: foo
)YAML",
                               "expected a value of scalar type."));
  // Not a string file in debug loc.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: { a: b }, Column: 12, Line: 12 }
)YAML",
                               "expected a value of scalar type."));
  // Not a integer column in debug loc.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Column: { a: b }, Line: 12 }
)YAML",
                               "expected a value of scalar type."));
  // Not a integer line in debug loc.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Column: 12, Line: { a: b } }
)YAML",
                               "expected a value of scalar type."));
  // Not a mapping type value for args.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
DebugLoc: { File: file.c, Column: 12, Line: { a: b } }
)YAML",
                               "expected a value of scalar type."));
}

TEST(OptRemarks, OptRemarksParsingWrongArgs) {
  // Multiple debug locs per arg.
  EXPECT_TRUE(
      parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
Args:
  - Str: string
    DebugLoc: { File: a, Line: 1, Column: 2 }
    DebugLoc: { File: a, Line: 1, Column: 2 }
)YAML",
                       "only one DebugLoc entry is allowed per argument."));
  // Multiple strings per arg.
  EXPECT_TRUE(
      parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
Args:
  - Str: string
    Str2: string
    DebugLoc: { File: a, Line: 1, Column: 2 }
)YAML",
                       "only one string entry is allowed per argument."));
  // No arg value.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
Args:
  - Callee: ''
  - DebugLoc: { File: a, Line: 1, Column: 2 }
)YAML",
                               "argument value is missing."));
  // No arg value.
  EXPECT_TRUE(parseExpectError(R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
Function: foo
Args:
  - DebugLoc: { File: a, Line: 1, Column: 2 }
)YAML",
                               "argument key is missing."));

}

TEST(OptRemarks, OptRemarksGoodStruct) {
  StringRef Buf = R"YAML(
--- !Missed
Pass: inline
Name: NoDefinition
DebugLoc: { File: file.c, Line: 3, Column: 12 }
Function: foo
Args:
  - Callee: bar
  - String: ' will not be inlined into '
  - Caller: foo
    DebugLoc: { File: file.c, Line: 2, Column: 0 }
  - String: ' because its definition is unavailable'
)YAML";

  LLVMOptRemarkParserRef Parser =
      LLVMOptRemarkParserCreate(Buf.data(), Buf.size());
  LLVMOptRemarkEntry *Remark = LLVMOptRemarkParserGetNext(Parser);
  EXPECT_FALSE(Remark == nullptr);
  EXPECT_EQ(StringRef(Remark->RemarkType.Str, 7), "!Missed");
  EXPECT_EQ(Remark->RemarkType.Len, 7U);
  EXPECT_EQ(StringRef(Remark->PassName.Str, 6), "inline");
  EXPECT_EQ(Remark->PassName.Len, 6U);
  EXPECT_EQ(StringRef(Remark->RemarkName.Str, 12), "NoDefinition");
  EXPECT_EQ(Remark->RemarkName.Len, 12U);
  EXPECT_EQ(StringRef(Remark->FunctionName.Str, 3), "foo");
  EXPECT_EQ(Remark->FunctionName.Len, 3U);
  EXPECT_EQ(StringRef(Remark->DebugLoc.SourceFile.Str, 6), "file.c");
  EXPECT_EQ(Remark->DebugLoc.SourceFile.Len, 6U);
  EXPECT_EQ(Remark->DebugLoc.SourceLineNumber, 3U);
  EXPECT_EQ(Remark->DebugLoc.SourceColumnNumber, 12U);
  EXPECT_EQ(Remark->Hotness, 0U);
  EXPECT_EQ(Remark->NumArgs, 4U);
  // Arg 0
  {
    LLVMOptRemarkArg &Arg = Remark->Args[0];
    EXPECT_EQ(StringRef(Arg.Key.Str, 6), "Callee");
    EXPECT_EQ(Arg.Key.Len, 6U);
    EXPECT_EQ(StringRef(Arg.Value.Str, 3), "bar");
    EXPECT_EQ(Arg.Value.Len, 3U);
    EXPECT_EQ(StringRef(Arg.DebugLoc.SourceFile.Str, 0), "");
    EXPECT_EQ(Arg.DebugLoc.SourceFile.Len, 0U);
    EXPECT_EQ(Arg.DebugLoc.SourceLineNumber, 0U);
    EXPECT_EQ(Arg.DebugLoc.SourceColumnNumber, 0U);
  }
  // Arg 1
  {
    LLVMOptRemarkArg &Arg = Remark->Args[1];
    EXPECT_EQ(StringRef(Arg.Key.Str, 6), "String");
    EXPECT_EQ(Arg.Key.Len, 6U);
    EXPECT_EQ(StringRef(Arg.Value.Str, 26), " will not be inlined into ");
    EXPECT_EQ(Arg.Value.Len, 26U);
    EXPECT_EQ(StringRef(Arg.DebugLoc.SourceFile.Str, 0), "");
    EXPECT_EQ(Arg.DebugLoc.SourceFile.Len, 0U);
    EXPECT_EQ(Arg.DebugLoc.SourceLineNumber, 0U);
    EXPECT_EQ(Arg.DebugLoc.SourceColumnNumber, 0U);
  }
  // Arg 2
  {
    LLVMOptRemarkArg &Arg = Remark->Args[2];
    EXPECT_EQ(StringRef(Arg.Key.Str, 6), "Caller");
    EXPECT_EQ(Arg.Key.Len, 6U);
    EXPECT_EQ(StringRef(Arg.Value.Str, 3), "foo");
    EXPECT_EQ(Arg.Value.Len, 3U);
    EXPECT_EQ(StringRef(Arg.DebugLoc.SourceFile.Str, 6), "file.c");
    EXPECT_EQ(Arg.DebugLoc.SourceFile.Len, 6U);
    EXPECT_EQ(Arg.DebugLoc.SourceLineNumber, 2U);
    EXPECT_EQ(Arg.DebugLoc.SourceColumnNumber, 0U);
  }
  // Arg 3
  {
    LLVMOptRemarkArg &Arg = Remark->Args[3];
    EXPECT_EQ(StringRef(Arg.Key.Str, 6), "String");
    EXPECT_EQ(Arg.Key.Len, 6U);
    EXPECT_EQ(StringRef(Arg.Value.Str, 38),
              " because its definition is unavailable");
    EXPECT_EQ(Arg.Value.Len, 38U);
    EXPECT_EQ(StringRef(Arg.DebugLoc.SourceFile.Str, 0), "");
    EXPECT_EQ(Arg.DebugLoc.SourceFile.Len, 0U);
    EXPECT_EQ(Arg.DebugLoc.SourceLineNumber, 0U);
    EXPECT_EQ(Arg.DebugLoc.SourceColumnNumber, 0U);
  }

  EXPECT_EQ(LLVMOptRemarkParserGetNext(Parser), nullptr);

  EXPECT_FALSE(LLVMOptRemarkParserHasError(Parser));
  LLVMOptRemarkParserDispose(Parser);
}
