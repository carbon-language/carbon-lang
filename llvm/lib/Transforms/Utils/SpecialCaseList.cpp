//===-- SpecialCaseList.cpp - special case list for sanitizers ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions or global
// variables, or to instrument some functions or global variables in a specific
// way, based on a user-supplied list.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SpecialCaseList.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <string>
#include <utility>

namespace llvm {

/// Represents a set of regular expressions.  Regular expressions which are
/// "literal" (i.e. no regex metacharacters) are stored in Strings, while all
/// others are represented as a single pipe-separated regex in RegEx.  The
/// reason for doing so is efficiency; StringSet is much faster at matching
/// literal strings than Regex.
struct SpecialCaseList::Entry {
  StringSet<> Strings;
  Regex *RegEx;

  Entry() : RegEx(0) {}

  bool match(StringRef Query) const {
    return Strings.count(Query) || (RegEx && RegEx->match(Query));
  }
};

SpecialCaseList::SpecialCaseList(const StringRef Path) {
  // Validate and open blacklist file.
  if (Path.empty()) return;
  OwningPtr<MemoryBuffer> File;
  if (error_code EC = MemoryBuffer::getFile(Path, File)) {
    report_fatal_error("Can't open blacklist file: " + Path + ": " +
                       EC.message());
  }

  init(File.get());
}

SpecialCaseList::SpecialCaseList(const MemoryBuffer *MB) {
  init(MB);
}

void SpecialCaseList::init(const MemoryBuffer *MB) {
  // Iterate through each line in the blacklist file.
  SmallVector<StringRef, 16> Lines;
  SplitString(MB->getBuffer(), Lines, "\n\r");
  StringMap<StringMap<std::string> > Regexps;
  for (SmallVectorImpl<StringRef>::iterator I = Lines.begin(), E = Lines.end();
       I != E; ++I) {
    // Ignore empty lines and lines starting with "#"
    if (I->empty() || I->startswith("#"))
      continue;
    // Get our prefix and unparsed regexp.
    std::pair<StringRef, StringRef> SplitLine = I->split(":");
    StringRef Prefix = SplitLine.first;
    if (SplitLine.second.empty()) {
      // Missing ':' in the line.
      report_fatal_error("malformed blacklist line: " + SplitLine.first);
    }

    std::pair<StringRef, StringRef> SplitRegexp = SplitLine.second.split("=");
    std::string Regexp = SplitRegexp.first;
    StringRef Category = SplitRegexp.second;

    // Backwards compatibility.
    if (Prefix == "global-init") {
      Prefix = "global";
      Category = "init";
    } else if (Prefix == "global-init-type") {
      Prefix = "type";
      Category = "init";
    } else if (Prefix == "global-init-src") {
      Prefix = "src";
      Category = "init";
    }

    // See if we can store Regexp in Strings.
    if (Regex::isLiteralERE(Regexp)) {
      Entries[Prefix][Category].Strings.insert(Regexp);
      continue;
    }

    // Replace * with .*
    for (size_t pos = 0; (pos = Regexp.find("*", pos)) != std::string::npos;
         pos += strlen(".*")) {
      Regexp.replace(pos, strlen("*"), ".*");
    }

    // Check that the regexp is valid.
    Regex CheckRE(Regexp);
    std::string Error;
    if (!CheckRE.isValid(Error)) {
      report_fatal_error("malformed blacklist regex: " + SplitLine.second +
          ": " + Error);
    }

    // Add this regexp into the proper group by its prefix.
    if (!Regexps[Prefix][Category].empty())
      Regexps[Prefix][Category] += "|";
    Regexps[Prefix][Category] += "^" + Regexp + "$";
  }

  // Iterate through each of the prefixes, and create Regexs for them.
  for (StringMap<StringMap<std::string> >::const_iterator I = Regexps.begin(),
                                                          E = Regexps.end();
       I != E; ++I) {
    for (StringMap<std::string>::const_iterator II = I->second.begin(),
                                                IE = I->second.end();
         II != IE; ++II) {
      Entries[I->getKey()][II->getKey()].RegEx = new Regex(II->getValue());
    }
  }
}

SpecialCaseList::~SpecialCaseList() {
  for (StringMap<StringMap<Entry> >::iterator I = Entries.begin(),
                                              E = Entries.end();
       I != E; ++I) {
    for (StringMap<Entry>::const_iterator II = I->second.begin(),
                                          IE = I->second.end();
         II != IE; ++II) {
      delete II->second.RegEx;
    }
  }
}

bool SpecialCaseList::findCategory(const Function &F,
                                   StringRef &Category) const {
  return findCategory(*F.getParent(), Category) ||
         findCategory("fun", F.getName(), Category);
}

bool SpecialCaseList::isIn(const Function& F, const StringRef Category) const {
  return isIn(*F.getParent(), Category) ||
         inSectionCategory("fun", F.getName(), Category);
}

static StringRef GetGVTypeString(const GlobalVariable &G) {
  // Types of GlobalVariables are always pointer types.
  Type *GType = G.getType()->getElementType();
  // For now we support blacklisting struct types only.
  if (StructType *SGType = dyn_cast<StructType>(GType)) {
    if (!SGType->isLiteral())
      return SGType->getName();
  }
  return "<unknown type>";
}

bool SpecialCaseList::findCategory(const GlobalVariable &G,
                                   StringRef &Category) const {
  return findCategory(*G.getParent(), Category) ||
         findCategory("global", G.getName(), Category) ||
         findCategory("type", GetGVTypeString(G), Category);
}

bool SpecialCaseList::isIn(const GlobalVariable &G,
                           const StringRef Category) const {
  return isIn(*G.getParent(), Category) ||
         inSectionCategory("global", G.getName(), Category) ||
         inSectionCategory("type", GetGVTypeString(G), Category);
}

bool SpecialCaseList::findCategory(const Module &M, StringRef &Category) const {
  return findCategory("src", M.getModuleIdentifier(), Category);
}

bool SpecialCaseList::isIn(const Module &M, const StringRef Category) const {
  return inSectionCategory("src", M.getModuleIdentifier(), Category);
}

bool SpecialCaseList::findCategory(const StringRef Section,
                                   const StringRef Query,
                                   StringRef &Category) const {
  StringMap<StringMap<Entry> >::const_iterator I = Entries.find(Section);
  if (I == Entries.end()) return false;

  for (StringMap<Entry>::const_iterator II = I->second.begin(),
                                        IE = I->second.end();
       II != IE; ++II) {
    if (II->getValue().match(Query)) {
      Category = II->first();
      return true;
    }
  }

  return false;
}

bool SpecialCaseList::inSectionCategory(const StringRef Section,
                                        const StringRef Query,
                                        const StringRef Category) const {
  StringMap<StringMap<Entry> >::const_iterator I = Entries.find(Section);
  if (I == Entries.end()) return false;
  StringMap<Entry>::const_iterator II = I->second.find(Category);
  if (II == I->second.end()) return false;

  return II->getValue().match(Query);
}

}  // namespace llvm
