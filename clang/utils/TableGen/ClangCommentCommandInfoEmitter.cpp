//===--- ClangCommentCommandInfoEmitter.cpp - Generate command lists -----====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits command lists and efficient matchers for command
// names that are used in documentation comments.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <vector>

using namespace llvm;

namespace clang {
void EmitClangCommentCommandInfo(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("A list of commands useable in documentation "
                       "comments", OS);

  OS << "namespace {\n"
        "const CommandInfo Commands[] = {\n";
  std::vector<Record *> Tags = Records.getAllDerivedDefinitions("Command");
  for (size_t i = 0, e = Tags.size(); i != e; ++i) {
    Record &Tag = *Tags[i];
    OS << "  { "
       << "\"" << Tag.getValueAsString("Name") << "\", "
       << "\"" << Tag.getValueAsString("EndCommandName") << "\", "
       << i << ", "
       << Tag.getValueAsInt("NumArgs") << ", "
       << Tag.getValueAsBit("IsInlineCommand") << ", "
       << Tag.getValueAsBit("IsBlockCommand") << ", "
       << Tag.getValueAsBit("IsBriefCommand") << ", "
       << Tag.getValueAsBit("IsReturnsCommand") << ", "
       << Tag.getValueAsBit("IsParamCommand") << ", "
       << Tag.getValueAsBit("IsTParamCommand") << ", "
       << Tag.getValueAsBit("IsDeprecatedCommand") << ", "
       << Tag.getValueAsBit("IsEmptyParagraphAllowed") << ", "
       << Tag.getValueAsBit("IsVerbatimBlockCommand") << ", "
       << Tag.getValueAsBit("IsVerbatimBlockEndCommand") << ", "
       << Tag.getValueAsBit("IsVerbatimLineCommand") << ", "
       << Tag.getValueAsBit("IsDeclarationCommand") << ", "
       << /* IsUnknownCommand = */ "0"
       << " }";
    if (i + 1 != e)
      OS << ",";
    OS << "\n";
  }
  OS << "};\n"
        "} // unnamed namespace\n\n";

  std::vector<StringMatcher::StringPair> Matches;
  for (size_t i = 0, e = Tags.size(); i != e; ++i) {
    Record &Tag = *Tags[i];
    std::string Name = Tag.getValueAsString("Name");
    std::string Return;
    raw_string_ostream(Return) << "return &Commands[" << i << "];";
    Matches.push_back(StringMatcher::StringPair(Name, Return));
  }

  OS << "const CommandInfo *CommandTraits::getBuiltinCommandInfo(\n"
     << "                                         StringRef Name) {\n";
  StringMatcher("Name", Matches, OS).Emit();
  OS << "  return NULL;\n"
     << "}\n\n";
}
} // end namespace clang

