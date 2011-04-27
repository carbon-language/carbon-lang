//===--- JsonCompileCommandLineDatabase.cpp - Simple JSON database --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements reading a compile command line database, as written
//  out for example by CMake.
//
//===----------------------------------------------------------------------===//

#include "JsonCompileCommandLineDatabase.h"
#include "llvm/ADT/Twine.h"

namespace clang {
namespace tooling {

namespace {

// A parser for JSON escaped strings of command line arguments with \-escaping
// for quoted arguments (see the documentation of UnescapeJsonCommandLine(...)).
class CommandLineArgumentParser {
 public:
  CommandLineArgumentParser(llvm::StringRef CommandLine)
      : Input(CommandLine), Position(Input.begin()-1) {}

  std::vector<std::string> Parse() {
    bool HasMoreInput = true;
    while (HasMoreInput && NextNonWhitespace()) {
      std::string Argument;
      HasMoreInput = ParseStringInto(Argument);
      CommandLine.push_back(Argument);
    }
    return CommandLine;
  }

 private:
  // All private methods return true if there is more input available.

  bool ParseStringInto(std::string &String) {
    do {
      if (*Position == '"') {
        if (!ParseQuotedStringInto(String)) return false;
      } else {
        if (!ParseFreeStringInto(String)) return false;
      }
    } while (*Position != ' ');
    return true;
  }

  bool ParseQuotedStringInto(std::string &String) {
    if (!Next()) return false;
    while (*Position != '"') {
      if (!SkipEscapeCharacter()) return false;
      String.push_back(*Position);
      if (!Next()) return false;
    }
    return Next();
  }

  bool ParseFreeStringInto(std::string &String) {
    do {
      if (!SkipEscapeCharacter()) return false;
      String.push_back(*Position);
      if (!Next()) return false;
    } while (*Position != ' ' && *Position != '"');
    return true;
  }

  bool SkipEscapeCharacter() {
    if (*Position == '\\') {
      return Next();
    }
    return true;
  }

  bool NextNonWhitespace() {
    do {
      if (!Next()) return false;
    } while (*Position == ' ');
    return true;
  }

  bool Next() {
    ++Position;
    if (Position == Input.end()) return false;
    // Remove the JSON escaping first. This is done unconditionally.
    if (*Position == '\\') ++Position;
    return Position != Input.end();
  }

  const llvm::StringRef Input;
  llvm::StringRef::iterator Position;
  std::vector<std::string> CommandLine;
};

} // end namespace

std::vector<std::string> UnescapeJsonCommandLine(
    llvm::StringRef JsonEscapedCommandLine) {
  CommandLineArgumentParser parser(JsonEscapedCommandLine);
  return parser.Parse();
}

JsonCompileCommandLineParser::JsonCompileCommandLineParser(
    const llvm::StringRef Input, CompileCommandHandler *CommandHandler)
    : Input(Input), Position(Input.begin()-1), CommandHandler(CommandHandler) {}

bool JsonCompileCommandLineParser::Parse() {
  NextNonWhitespace();
  return ParseTranslationUnits();
}

std::string JsonCompileCommandLineParser::GetErrorMessage() const {
  return ErrorMessage;
}

bool JsonCompileCommandLineParser::ParseTranslationUnits() {
  if (!ConsumeOrError('[', "at start of compile command file")) return false;
  if (!ParseTranslationUnit(/*First=*/true)) return false;
  while (Consume(',')) {
    if (!ParseTranslationUnit(/*First=*/false)) return false;
  }
  if (!ConsumeOrError(']', "at end of array")) return false;
  if (CommandHandler != NULL) {
    CommandHandler->EndTranslationUnits();
  }
  return true;
}

bool JsonCompileCommandLineParser::ParseTranslationUnit(bool First) {
  if (First) {
    if (!Consume('{')) return true;
  } else {
    if (!ConsumeOrError('{', "at start of object")) return false;
  }
  if (!Consume('}')) {
    if (!ParseObjectKeyValuePairs()) return false;
    if (!ConsumeOrError('}', "at end of object")) return false;
  }
  if (CommandHandler != NULL) {
    CommandHandler->EndTranslationUnit();
  }
  return true;
}

bool JsonCompileCommandLineParser::ParseObjectKeyValuePairs() {
  do {
    llvm::StringRef Key;
    if (!ParseString(Key)) return false;
    if (!ConsumeOrError(':', "between name and value")) return false;
    llvm::StringRef Value;
    if (!ParseString(Value)) return false;
    if (CommandHandler != NULL) {
      CommandHandler->HandleKeyValue(Key, Value);
    }
  } while (Consume(','));
  return true;
}

bool JsonCompileCommandLineParser::ParseString(llvm::StringRef &String) {
  if (!ConsumeOrError('"', "at start of string")) return false;
  llvm::StringRef::iterator First = Position;
  llvm::StringRef::iterator Last = Position;
  while (!Consume('"')) {
    Consume('\\');
    ++Position;
    // We need to store Position, as Consume will change Last before leaving
    // the loop.
    Last = Position;
  }
  String = llvm::StringRef(First, Last - First);
  return true;
}

bool JsonCompileCommandLineParser::Consume(char C) {
  if (Position == Input.end()) return false;
  if (*Position != C) return false;
  NextNonWhitespace();
  return true;
}

bool JsonCompileCommandLineParser::ConsumeOrError(
    char C, llvm::StringRef Message) {
  if (!Consume(C)) {
    SetExpectError(C, Message);
    return false;
  }
  return true;
}

void JsonCompileCommandLineParser::SetExpectError(
    char C, llvm::StringRef Message) {
  ErrorMessage = (llvm::Twine("'") + llvm::StringRef(&C, 1) +
                  "' expected " + Message + ".").str();
}

void JsonCompileCommandLineParser::NextNonWhitespace() {
  do {
    ++Position;
  } while (IsWhitespace());
}

bool JsonCompileCommandLineParser::IsWhitespace() {
  if (Position == Input.end()) return false;
  return (*Position == ' ' || *Position == '\t' ||
          *Position == '\n' || *Position == '\r');
}

} // end namespace tooling
} // end namespace clang
