//===- unittest/Tooling/CompilationDatabaseTest.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

static void expectFailure(StringRef JSONDatabase, StringRef Explanation) {
  std::string ErrorMessage;
  EXPECT_EQ(NULL, JSONCompilationDatabase::loadFromBuffer(JSONDatabase,
                                                          ErrorMessage))
    << "Expected an error because of: " << Explanation;
}

TEST(JSONCompilationDatabase, ErrsOnInvalidFormat) {
  expectFailure("", "Empty database");
  expectFailure("{", "Invalid JSON");
  expectFailure("[[]]", "Array instead of object");
  expectFailure("[{\"a\":[]}]", "Array instead of value");
  expectFailure("[{\"a\":\"b\"}]", "Unknown key");
  expectFailure("[{[]:\"\"}]", "Incorrectly typed entry");
  expectFailure("[{}]", "Empty entry");
  expectFailure("[{\"directory\":\"\",\"command\":\"\"}]", "Missing file");
  expectFailure("[{\"directory\":\"\",\"file\":\"\"}]", "Missing command");
  expectFailure("[{\"command\":\"\",\"file\":\"\"}]", "Missing directory");
}

static CompileCommand findCompileArgsInJsonDatabase(StringRef FileName,
                                                    StringRef JSONDatabase,
                                                    std::string &ErrorMessage) {
  llvm::OwningPtr<CompilationDatabase> Database(
      JSONCompilationDatabase::loadFromBuffer(JSONDatabase, ErrorMessage));
  if (!Database)
    return CompileCommand();
  std::vector<CompileCommand> Commands = Database->getCompileCommands(FileName);
  EXPECT_LE(Commands.size(), 1u);
  if (Commands.empty())
    return CompileCommand();
  return Commands[0];
}

TEST(findCompileArgsInJsonDatabase, FindsNothingIfEmpty) {
  std::string ErrorMessage;
  CompileCommand NotFound = findCompileArgsInJsonDatabase(
    "a-file.cpp", "", ErrorMessage);
  EXPECT_TRUE(NotFound.CommandLine.empty()) << ErrorMessage;
  EXPECT_TRUE(NotFound.Directory.empty()) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ReadsSingleEntry) {
  StringRef Directory("/some/directory");
  StringRef FileName("/path/to/a-file.cpp");
  StringRef Command("/path/to/compiler and some arguments");
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    FileName,
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  EXPECT_EQ(Directory, FoundCommand.Directory) << ErrorMessage;
  ASSERT_EQ(4u, FoundCommand.CommandLine.size()) << ErrorMessage;
  EXPECT_EQ("/path/to/compiler", FoundCommand.CommandLine[0]) << ErrorMessage;
  EXPECT_EQ("and", FoundCommand.CommandLine[1]) << ErrorMessage;
  EXPECT_EQ("some", FoundCommand.CommandLine[2]) << ErrorMessage;
  EXPECT_EQ("arguments", FoundCommand.CommandLine[3]) << ErrorMessage;

  CompileCommand NotFound = findCompileArgsInJsonDatabase(
    "a-file.cpp",
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  EXPECT_TRUE(NotFound.Directory.empty()) << ErrorMessage;
  EXPECT_TRUE(NotFound.CommandLine.empty()) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ReadsCompileCommandLinesWithSpaces) {
  StringRef Directory("/some/directory");
  StringRef FileName("/path/to/a-file.cpp");
  StringRef Command("\\\"/path to compiler\\\" \\\"and an argument\\\"");
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    FileName,
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  ASSERT_EQ(2u, FoundCommand.CommandLine.size());
  EXPECT_EQ("/path to compiler", FoundCommand.CommandLine[0]) << ErrorMessage;
  EXPECT_EQ("and an argument", FoundCommand.CommandLine[1]) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ReadsDirectoryWithSpaces) {
  StringRef Directory("/some directory / with spaces");
  StringRef FileName("/path/to/a-file.cpp");
  StringRef Command("a command");
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    FileName,
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  EXPECT_EQ(Directory, FoundCommand.Directory) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, FindsEntry) {
  StringRef Directory("directory");
  StringRef FileName("file");
  StringRef Command("command");
  std::string JsonDatabase = "[";
  for (int I = 0; I < 10; ++I) {
    if (I > 0) JsonDatabase += ",";
    JsonDatabase +=
      ("{\"directory\":\"" + Directory + Twine(I) + "\"," +
        "\"command\":\"" + Command + Twine(I) + "\","
        "\"file\":\"" + FileName + Twine(I) + "\"}").str();
  }
  JsonDatabase += "]";
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    "file4", JsonDatabase, ErrorMessage);
  EXPECT_EQ("directory4", FoundCommand.Directory) << ErrorMessage;
  ASSERT_EQ(1u, FoundCommand.CommandLine.size()) << ErrorMessage;
  EXPECT_EQ("command4", FoundCommand.CommandLine[0]) << ErrorMessage;
}

static std::vector<std::string> unescapeJsonCommandLine(StringRef Command) {
  std::string JsonDatabase =
    ("[{\"directory\":\"\", \"file\":\"test\", \"command\": \"" +
     Command + "\"}]").str();
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    "test", JsonDatabase, ErrorMessage);
  EXPECT_TRUE(ErrorMessage.empty()) << ErrorMessage;
  return FoundCommand.CommandLine;
}

TEST(unescapeJsonCommandLine, ReturnsEmptyArrayOnEmptyString) {
  std::vector<std::string> Result = unescapeJsonCommandLine("");
  EXPECT_TRUE(Result.empty());
}

TEST(unescapeJsonCommandLine, SplitsOnSpaces) {
  std::vector<std::string> Result = unescapeJsonCommandLine("a b c");
  ASSERT_EQ(3ul, Result.size());
  EXPECT_EQ("a", Result[0]);
  EXPECT_EQ("b", Result[1]);
  EXPECT_EQ("c", Result[2]);
}

TEST(unescapeJsonCommandLine, MungesMultipleSpaces) {
  std::vector<std::string> Result = unescapeJsonCommandLine("   a   b   ");
  ASSERT_EQ(2ul, Result.size());
  EXPECT_EQ("a", Result[0]);
  EXPECT_EQ("b", Result[1]);
}

TEST(unescapeJsonCommandLine, UnescapesBackslashCharacters) {
  std::vector<std::string> Backslash = unescapeJsonCommandLine("a\\\\\\\\");
  ASSERT_EQ(1ul, Backslash.size());
  EXPECT_EQ("a\\", Backslash[0]);
  std::vector<std::string> Quote = unescapeJsonCommandLine("a\\\\\\\"");
  ASSERT_EQ(1ul, Quote.size());
  EXPECT_EQ("a\"", Quote[0]);
}

TEST(unescapeJsonCommandLine, DoesNotMungeSpacesBetweenQuotes) {
  std::vector<std::string> Result = unescapeJsonCommandLine("\\\"  a  b  \\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("  a  b  ", Result[0]);
}

TEST(unescapeJsonCommandLine, AllowsMultipleQuotedArguments) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "  \\\" a \\\"  \\\" b \\\"  ");
  ASSERT_EQ(2ul, Result.size());
  EXPECT_EQ(" a ", Result[0]);
  EXPECT_EQ(" b ", Result[1]);
}

TEST(unescapeJsonCommandLine, AllowsEmptyArgumentsInQuotes) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "\\\"\\\"\\\"\\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_TRUE(Result[0].empty()) << Result[0];
}

TEST(unescapeJsonCommandLine, ParsesEscapedQuotesInQuotedStrings) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "\\\"\\\\\\\"\\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("\"", Result[0]);
}

TEST(unescapeJsonCommandLine, ParsesMultipleArgumentsWithEscapedCharacters) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "  \\\\\\\"  \\\"a \\\\\\\" b \\\"     \\\"and\\\\\\\\c\\\"   \\\\\\\"");
  ASSERT_EQ(4ul, Result.size());
  EXPECT_EQ("\"", Result[0]);
  EXPECT_EQ("a \" b ", Result[1]);
  EXPECT_EQ("and\\c", Result[2]);
  EXPECT_EQ("\"", Result[3]);
}

TEST(unescapeJsonCommandLine, ParsesStringsWithoutSpacesIntoSingleArgument) {
  std::vector<std::string> QuotedNoSpaces = unescapeJsonCommandLine(
      "\\\"a\\\"\\\"b\\\"");
  ASSERT_EQ(1ul, QuotedNoSpaces.size());
  EXPECT_EQ("ab", QuotedNoSpaces[0]);

  std::vector<std::string> MixedNoSpaces = unescapeJsonCommandLine(
      "\\\"a\\\"bcd\\\"ef\\\"\\\"\\\"\\\"g\\\"");
  ASSERT_EQ(1ul, MixedNoSpaces.size());
  EXPECT_EQ("abcdefg", MixedNoSpaces[0]);
}

TEST(unescapeJsonCommandLine, ParsesQuotedStringWithoutClosingQuote) {
  std::vector<std::string> Unclosed = unescapeJsonCommandLine("\\\"abc");
  ASSERT_EQ(1ul, Unclosed.size());
  EXPECT_EQ("abc", Unclosed[0]);

  std::vector<std::string> Empty = unescapeJsonCommandLine("\\\"");
  ASSERT_EQ(1ul, Empty.size());
  EXPECT_EQ("", Empty[0]);
}

TEST(FixedCompilationDatabase, ReturnsFixedCommandLine) {
  std::vector<std::string> CommandLine;
  CommandLine.push_back("one");
  CommandLine.push_back("two");
  FixedCompilationDatabase Database(".", CommandLine);
  std::vector<CompileCommand> Result =
    Database.getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  std::vector<std::string> ExpectedCommandLine(1, "clang-tool");
  ExpectedCommandLine.insert(ExpectedCommandLine.end(),
                             CommandLine.begin(), CommandLine.end());
  ExpectedCommandLine.push_back("source");
  EXPECT_EQ(".", Result[0].Directory);
  EXPECT_EQ(ExpectedCommandLine, Result[0].CommandLine);
}

TEST(ParseFixedCompilationDatabase, ReturnsNullOnEmptyArgumentList) {
  int Argc = 0;
  llvm::OwningPtr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, NULL));
  EXPECT_FALSE(Database);
  EXPECT_EQ(0, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsNullWithoutDoubleDash) {
  int Argc = 2;
  const char *Argv[] = { "1", "2" };
  llvm::OwningPtr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  EXPECT_FALSE(Database);
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsArgumentsAfterDoubleDash) {
  int Argc = 5;
  const char *Argv[] = { "1", "2", "--\0no-constant-folding", "3", "4" };
  llvm::OwningPtr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  ASSERT_TRUE(Database);
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> CommandLine;
  CommandLine.push_back("clang-tool");
  CommandLine.push_back("3");
  CommandLine.push_back("4");
  CommandLine.push_back("source");
  ASSERT_EQ(CommandLine, Result[0].CommandLine);
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsEmptyCommandLine) {
  int Argc = 3;
  const char *Argv[] = { "1", "2", "--\0no-constant-folding" };
  llvm::OwningPtr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  ASSERT_TRUE(Database);
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> CommandLine;
  CommandLine.push_back("clang-tool");
  CommandLine.push_back("source");
  ASSERT_EQ(CommandLine, Result[0].CommandLine);
  EXPECT_EQ(2, Argc);
}

} // end namespace tooling
} // end namespace clang
