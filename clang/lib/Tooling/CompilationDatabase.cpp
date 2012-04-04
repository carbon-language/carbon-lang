//===--- CompilationDatabase.cpp - ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains multiple implementations for CompilationDatabases.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSONParser.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/system_error.h"

namespace clang {
namespace tooling {

namespace {

/// \brief A parser for JSON escaped strings of command line arguments.
///
/// Assumes \-escaping for quoted arguments (see the documentation of
/// unescapeJSONCommandLine(...)).
class CommandLineArgumentParser {
 public:
  CommandLineArgumentParser(StringRef CommandLine)
      : Input(CommandLine), Position(Input.begin()-1) {}

  std::vector<std::string> parse() {
    bool HasMoreInput = true;
    while (HasMoreInput && nextNonWhitespace()) {
      std::string Argument;
      HasMoreInput = parseStringInto(Argument);
      CommandLine.push_back(Argument);
    }
    return CommandLine;
  }

 private:
  // All private methods return true if there is more input available.

  bool parseStringInto(std::string &String) {
    do {
      if (*Position == '"') {
        if (!parseQuotedStringInto(String)) return false;
      } else {
        if (!parseFreeStringInto(String)) return false;
      }
    } while (*Position != ' ');
    return true;
  }

  bool parseQuotedStringInto(std::string &String) {
    if (!next()) return false;
    while (*Position != '"') {
      if (!skipEscapeCharacter()) return false;
      String.push_back(*Position);
      if (!next()) return false;
    }
    return next();
  }

  bool parseFreeStringInto(std::string &String) {
    do {
      if (!skipEscapeCharacter()) return false;
      String.push_back(*Position);
      if (!next()) return false;
    } while (*Position != ' ' && *Position != '"');
    return true;
  }

  bool skipEscapeCharacter() {
    if (*Position == '\\') {
      return next();
    }
    return true;
  }

  bool nextNonWhitespace() {
    do {
      if (!next()) return false;
    } while (*Position == ' ');
    return true;
  }

  bool next() {
    ++Position;
    if (Position == Input.end()) return false;
    // Remove the JSON escaping first. This is done unconditionally.
    if (*Position == '\\') ++Position;
    return Position != Input.end();
  }

  const StringRef Input;
  StringRef::iterator Position;
  std::vector<std::string> CommandLine;
};

std::vector<std::string> unescapeJSONCommandLine(
    StringRef JSONEscapedCommandLine) {
  CommandLineArgumentParser parser(JSONEscapedCommandLine);
  return parser.parse();
}

} // end namespace

CompilationDatabase::~CompilationDatabase() {}

CompilationDatabase *
CompilationDatabase::loadFromDirectory(StringRef BuildDirectory,
                                       std::string &ErrorMessage) {
  llvm::SmallString<1024> JSONDatabasePath(BuildDirectory);
  llvm::sys::path::append(JSONDatabasePath, "compile_commands.json");
  llvm::OwningPtr<CompilationDatabase> Database(
    JSONCompilationDatabase::loadFromFile(JSONDatabasePath, ErrorMessage));
  if (!Database) {
    return NULL;
  }
  return Database.take();
}

JSONCompilationDatabase *
JSONCompilationDatabase::loadFromFile(StringRef FilePath,
                                      std::string &ErrorMessage) {
  llvm::OwningPtr<llvm::MemoryBuffer> DatabaseBuffer;
  llvm::error_code Result =
    llvm::MemoryBuffer::getFile(FilePath, DatabaseBuffer);
  if (Result != 0) {
    ErrorMessage = "Error while opening JSON database: " + Result.message();
    return NULL;
  }
  llvm::OwningPtr<JSONCompilationDatabase> Database(
    new JSONCompilationDatabase(DatabaseBuffer.take()));
  if (!Database->parse(ErrorMessage))
    return NULL;
  return Database.take();
}

JSONCompilationDatabase *
JSONCompilationDatabase::loadFromBuffer(StringRef DatabaseString,
                                        std::string &ErrorMessage) {
  llvm::OwningPtr<llvm::MemoryBuffer> DatabaseBuffer(
      llvm::MemoryBuffer::getMemBuffer(DatabaseString));
  llvm::OwningPtr<JSONCompilationDatabase> Database(
    new JSONCompilationDatabase(DatabaseBuffer.take()));
  if (!Database->parse(ErrorMessage))
    return NULL;
  return Database.take();
}

std::vector<CompileCommand>
JSONCompilationDatabase::getCompileCommands(StringRef FilePath) const {
  llvm::StringMap< std::vector<CompileCommandRef> >::const_iterator
    CommandsRefI = IndexByFile.find(FilePath);
  if (CommandsRefI == IndexByFile.end())
    return std::vector<CompileCommand>();
  const std::vector<CompileCommandRef> &CommandsRef = CommandsRefI->getValue();
  std::vector<CompileCommand> Commands;
  for (int I = 0, E = CommandsRef.size(); I != E; ++I) {
    Commands.push_back(CompileCommand(
      // FIXME: Escape correctly:
      CommandsRef[I].first,
      unescapeJSONCommandLine(CommandsRef[I].second)));
  }
  return Commands;
}

bool JSONCompilationDatabase::parse(std::string &ErrorMessage) {
  llvm::SourceMgr SM;
  llvm::JSONParser Parser(Database->getBuffer(), &SM);
  llvm::JSONValue *Root = Parser.parseRoot();
  if (Root == NULL) {
    ErrorMessage = "Error while parsing JSON.";
    return false;
  }
  llvm::JSONArray *Array = dyn_cast<llvm::JSONArray>(Root);
  if (Array == NULL) {
    ErrorMessage = "Expected array.";
    return false;
  }
  for (llvm::JSONArray::const_iterator AI = Array->begin(), AE = Array->end();
       AI != AE; ++AI) {
    const llvm::JSONObject *Object = dyn_cast<llvm::JSONObject>(*AI);
    if (Object == NULL) {
      ErrorMessage = "Expected object.";
      return false;
    }
    StringRef EntryDirectory;
    StringRef EntryFile;
    StringRef EntryCommand;
    for (llvm::JSONObject::const_iterator KVI = Object->begin(),
                                          KVE = Object->end();
         KVI != KVE; ++KVI) {
      const llvm::JSONValue *Value = (*KVI)->Value;
      if (Value == NULL) {
        ErrorMessage = "Expected value.";
        return false;
      }
      const llvm::JSONString *ValueString =
        dyn_cast<llvm::JSONString>(Value);
      if (ValueString == NULL) {
        ErrorMessage = "Expected string as value.";
        return false;
      }
      if ((*KVI)->Key->getRawText() == "directory") {
        EntryDirectory = ValueString->getRawText();
      } else if ((*KVI)->Key->getRawText() == "file") {
        EntryFile = ValueString->getRawText();
      } else if ((*KVI)->Key->getRawText() == "command") {
        EntryCommand = ValueString->getRawText();
      } else {
        ErrorMessage = (Twine("Unknown key: \"") +
                        (*KVI)->Key->getRawText() + "\"").str();
        return false;
      }
    }
    IndexByFile[EntryFile].push_back(
      CompileCommandRef(EntryDirectory, EntryCommand));
  }
  return true;
}

} // end namespace tooling
} // end namespace clang

