//===- FileCheck.cpp - Check that File's Contents match what is expected --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FileCheck does a line-by line check of a file that validates whether it
// contains the expected content.  This is useful for regression tests etc.
//
// This program exits with an error status of 2 on error, exit status of 0 if
// the file matched the expected contents, and exit status of 1 if it did not
// contain the expected contents.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
using namespace llvm;

static cl::opt<std::string>
CheckFilename(cl::Positional, cl::desc("<check-file>"), cl::Required);

static cl::opt<std::string>
InputFilename("input-file", cl::desc("File to check (defaults to stdin)"),
              cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
CheckPrefix("check-prefix", cl::init("CHECK"),
            cl::desc("Prefix to use from check file (defaults to 'CHECK')"));

static cl::opt<bool>
NoCanonicalizeWhiteSpace("strict-whitespace",
              cl::desc("Do not treat all horizontal whitespace as equivalent"));

//===----------------------------------------------------------------------===//
// Pattern Handling Code.
//===----------------------------------------------------------------------===//

class Pattern {
  SourceMgr *SM;
  SMLoc PatternLoc;
  
  /// FixedStr - If non-empty, this pattern is a fixed string match with the
  /// specified fixed string.
  StringRef FixedStr;
  
  /// RegEx - If non-empty, this is a regex pattern.
  std::string RegExStr;
public:
  
  Pattern() { }
  
  bool ParsePattern(StringRef PatternStr, SourceMgr &SM);
  
  /// Match - Match the pattern string against the input buffer Buffer.  This
  /// returns the position that is matched or npos if there is no match.  If
  /// there is a match, the size of the matched string is returned in MatchLen.
  size_t Match(StringRef Buffer, size_t &MatchLen) const;
  
private:
  void AddFixedStringToRegEx(StringRef FixedStr);
};

bool Pattern::ParsePattern(StringRef PatternStr, SourceMgr &SM) {
  this->SM = &SM;
  PatternLoc = SMLoc::getFromPointer(PatternStr.data());
  
  // Ignore trailing whitespace.
  while (!PatternStr.empty() &&
         (PatternStr.back() == ' ' || PatternStr.back() == '\t'))
    PatternStr = PatternStr.substr(0, PatternStr.size()-1);
  
  // Check that there is something on the line.
  if (PatternStr.empty()) {
    SM.PrintMessage(PatternLoc, "found empty check string with prefix '" +
                    CheckPrefix+":'", "error");
    return true;
  }
  
  // Check to see if this is a fixed string, or if it has regex pieces.
  if (PatternStr.size() < 2 || PatternStr.find("{{") == StringRef::npos) {
    FixedStr = PatternStr;
    return false;
  }
  
  // Otherwise, there is at least one regex piece.  Build up the regex pattern
  // by escaping scary characters in fixed strings, building up one big regex.
  while (!PatternStr.empty()) {
    // Handle fixed string matches.
    if (PatternStr.size() < 2 ||
        PatternStr[0] != '{' || PatternStr[1] != '{') {
      // Find the end, which is the start of the next regex.
      size_t FixedMatchEnd = PatternStr.find("{{");
      AddFixedStringToRegEx(PatternStr.substr(0, FixedMatchEnd));
      PatternStr = PatternStr.substr(FixedMatchEnd);
      continue;
    }
    
    // Otherwise, this is the start of a regex match.  Scan for the }}.
    size_t End = PatternStr.find("}}");
    if (End == StringRef::npos) {
      SM.PrintMessage(SMLoc::getFromPointer(PatternStr.data()),
                      "found start of regex string with no end '}}'", "error");
      return true;
    }
    
    StringRef RegexStr = PatternStr.substr(2, End-2);
    Regex R(RegexStr);
    std::string Error;
    if (!R.isValid(Error)) {
      SM.PrintMessage(SMLoc::getFromPointer(PatternStr.data()+2),
                      "invalid regex: " + Error, "error");
      return true;
    }
    
    RegExStr += RegexStr.str();
    PatternStr = PatternStr.substr(End+2);
  }

  return false;
}

void Pattern::AddFixedStringToRegEx(StringRef FixedStr) {
  // Add the characters from FixedStr to the regex, escaping as needed.  This
  // avoids "leaning toothpicks" in common patterns.
  for (unsigned i = 0, e = FixedStr.size(); i != e; ++i) {
    switch (FixedStr[i]) {
    // These are the special characters matched in "p_ere_exp".
    case '(':
    case ')':
    case '^':
    case '$':
    case '|':
    case '*':
    case '+':
    case '?':
    case '.':
    case '[':
    case '\\':
    case '{':
      RegExStr += '\\';
      // FALL THROUGH.
    default:
      RegExStr += FixedStr[i];
      break;
    }
  }
}


/// Match - Match the pattern string against the input buffer Buffer.  This
/// returns the position that is matched or npos if there is no match.  If
/// there is a match, the size of the matched string is returned in MatchLen.
size_t Pattern::Match(StringRef Buffer, size_t &MatchLen) const {
  // If this is a fixed string pattern, just match it now.
  if (!FixedStr.empty()) {
    MatchLen = FixedStr.size();
    return Buffer.find(FixedStr);
  }
  
  // Regex match.
  SmallVector<StringRef, 4> MatchInfo;
  if (!Regex(RegExStr, Regex::Sub|Regex::Newline).match(Buffer, &MatchInfo))
    return StringRef::npos;
  
  // Successful regex match.
  assert(!MatchInfo.empty() && "Didn't get any match");
  StringRef FullMatch = MatchInfo[0];
  
  
  if (MatchInfo.size() != 1) {
    SM->PrintMessage(PatternLoc, "regex cannot use grouping parens", "error");
    exit(1);
  }
    
  
  MatchLen = FullMatch.size();
  return FullMatch.data()-Buffer.data();
}


//===----------------------------------------------------------------------===//
// Check Strings.
//===----------------------------------------------------------------------===//

/// CheckString - This is a check that we found in the input file.
struct CheckString {
  /// Pat - The pattern to match.
  Pattern Pat;
  
  /// Loc - The location in the match file that the check string was specified.
  SMLoc Loc;
  
  /// IsCheckNext - This is true if this is a CHECK-NEXT: directive (as opposed
  /// to a CHECK: directive.
  bool IsCheckNext;
  
  /// NotStrings - These are all of the strings that are disallowed from
  /// occurring between this match string and the previous one (or start of
  /// file).
  std::vector<std::pair<SMLoc, Pattern> > NotStrings;
  
  CheckString(const Pattern &P, SMLoc L, bool isCheckNext)
    : Pat(P), Loc(L), IsCheckNext(isCheckNext) {}
};

/// CanonicalizeInputFile - Remove duplicate horizontal space from the specified
/// memory buffer, free it, and return a new one.
static MemoryBuffer *CanonicalizeInputFile(MemoryBuffer *MB) {
  SmallVector<char, 16> NewFile;
  NewFile.reserve(MB->getBufferSize());
  
  for (const char *Ptr = MB->getBufferStart(), *End = MB->getBufferEnd();
       Ptr != End; ++Ptr) {
    // If C is not a horizontal whitespace, skip it.
    if (*Ptr != ' ' && *Ptr != '\t') {
      NewFile.push_back(*Ptr);
      continue;
    }
    
    // Otherwise, add one space and advance over neighboring space.
    NewFile.push_back(' ');
    while (Ptr+1 != End &&
           (Ptr[1] == ' ' || Ptr[1] == '\t'))
      ++Ptr;
  }
  
  // Free the old buffer and return a new one.
  MemoryBuffer *MB2 =
    MemoryBuffer::getMemBufferCopy(NewFile.data(), 
                                   NewFile.data() + NewFile.size(),
                                   MB->getBufferIdentifier());
  
  delete MB;
  return MB2;
}


/// ReadCheckFile - Read the check file, which specifies the sequence of
/// expected strings.  The strings are added to the CheckStrings vector.
static bool ReadCheckFile(SourceMgr &SM,
                          std::vector<CheckString> &CheckStrings) {
  // Open the check file, and tell SourceMgr about it.
  std::string ErrorStr;
  MemoryBuffer *F =
    MemoryBuffer::getFileOrSTDIN(CheckFilename.c_str(), &ErrorStr);
  if (F == 0) {
    errs() << "Could not open check file '" << CheckFilename << "': " 
           << ErrorStr << '\n';
    return true;
  }
  
  // If we want to canonicalize whitespace, strip excess whitespace from the
  // buffer containing the CHECK lines.
  if (!NoCanonicalizeWhiteSpace)
    F = CanonicalizeInputFile(F);
  
  SM.AddNewSourceBuffer(F, SMLoc());

  // Find all instances of CheckPrefix followed by : in the file.
  StringRef Buffer = F->getBuffer();

  std::vector<std::pair<SMLoc, Pattern> > NotMatches;
  
  while (1) {
    // See if Prefix occurs in the memory buffer.
    Buffer = Buffer.substr(Buffer.find(CheckPrefix));
    
    // If we didn't find a match, we're done.
    if (Buffer.empty())
      break;
    
    const char *CheckPrefixStart = Buffer.data();
    
    // When we find a check prefix, keep track of whether we find CHECK: or
    // CHECK-NEXT:
    bool IsCheckNext = false, IsCheckNot = false;
    
    // Verify that the : is present after the prefix.
    if (Buffer[CheckPrefix.size()] == ':') {
      Buffer = Buffer.substr(CheckPrefix.size()+1);
    } else if (Buffer.size() > CheckPrefix.size()+6 &&
               memcmp(Buffer.data()+CheckPrefix.size(), "-NEXT:", 6) == 0) {
      Buffer = Buffer.substr(CheckPrefix.size()+7);
      IsCheckNext = true;
    } else if (Buffer.size() > CheckPrefix.size()+5 &&
               memcmp(Buffer.data()+CheckPrefix.size(), "-NOT:", 5) == 0) {
      Buffer = Buffer.substr(CheckPrefix.size()+6);
      IsCheckNot = true;
    } else {
      Buffer = Buffer.substr(1);
      continue;
    }
    
    // Okay, we found the prefix, yay.  Remember the rest of the line, but
    // ignore leading and trailing whitespace.
    Buffer = Buffer.substr(Buffer.find_first_not_of(" \t"));
    
    // Scan ahead to the end of line.
    size_t EOL = Buffer.find_first_of("\n\r");

    // Parse the pattern.
    Pattern P;
    if (P.ParsePattern(Buffer.substr(0, EOL), SM))
      return true;
    
    Buffer = Buffer.substr(EOL);

    
    // Verify that CHECK-NEXT lines have at least one CHECK line before them.
    if (IsCheckNext && CheckStrings.empty()) {
      SM.PrintMessage(SMLoc::getFromPointer(CheckPrefixStart),
                      "found '"+CheckPrefix+"-NEXT:' without previous '"+
                      CheckPrefix+ ": line", "error");
      return true;
    }
    
    // Handle CHECK-NOT.
    if (IsCheckNot) {
      NotMatches.push_back(std::make_pair(SMLoc::getFromPointer(Buffer.data()),
                                          P));
      continue;
    }
    
    
    // Okay, add the string we captured to the output vector and move on.
    CheckStrings.push_back(CheckString(P,
                                       SMLoc::getFromPointer(Buffer.data()),
                                       IsCheckNext));
    std::swap(NotMatches, CheckStrings.back().NotStrings);
  }
  
  if (CheckStrings.empty()) {
    errs() << "error: no check strings found with prefix '" << CheckPrefix
           << ":'\n";
    return true;
  }
  
  if (!NotMatches.empty()) {
    errs() << "error: '" << CheckPrefix
           << "-NOT:' not supported after last check line.\n";
    return true;
  }
  
  return false;
}

static void PrintCheckFailed(const SourceMgr &SM, const CheckString &CheckStr,
                             StringRef Buffer) {
  // Otherwise, we have an error, emit an error message.
  SM.PrintMessage(CheckStr.Loc, "expected string not found in input",
                  "error");
  
  // Print the "scanning from here" line.  If the current position is at the
  // end of a line, advance to the start of the next line.
  Buffer = Buffer.substr(Buffer.find_first_not_of(" \t\n\r"));
  
  SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()), "scanning from here",
                  "note");
}

/// CountNumNewlinesBetween - Count the number of newlines in the specified
/// range.
static unsigned CountNumNewlinesBetween(StringRef Range) {
  unsigned NumNewLines = 0;
  while (1) {
    // Scan for newline.
    Range = Range.substr(Range.find_first_of("\n\r"));
    if (Range.empty()) return NumNewLines;
    
    ++NumNewLines;
    
    // Handle \n\r and \r\n as a single newline.
    if (Range.size() > 1 &&
        (Range[1] == '\n' || Range[1] == '\r') &&
        (Range[0] != Range[1]))
      Range = Range.substr(1);
    Range = Range.substr(1);
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  SourceMgr SM;
  
  // Read the expected strings from the check file.
  std::vector<CheckString> CheckStrings;
  if (ReadCheckFile(SM, CheckStrings))
    return 2;

  // Open the file to check and add it to SourceMgr.
  std::string ErrorStr;
  MemoryBuffer *F =
    MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), &ErrorStr);
  if (F == 0) {
    errs() << "Could not open input file '" << InputFilename << "': " 
           << ErrorStr << '\n';
    return true;
  }
  
  // Remove duplicate spaces in the input file if requested.
  if (!NoCanonicalizeWhiteSpace)
    F = CanonicalizeInputFile(F);
  
  SM.AddNewSourceBuffer(F, SMLoc());
  
  // Check that we have all of the expected strings, in order, in the input
  // file.
  StringRef Buffer = F->getBuffer();
  
  const char *LastMatch = Buffer.data();
  
  for (unsigned StrNo = 0, e = CheckStrings.size(); StrNo != e; ++StrNo) {
    const CheckString &CheckStr = CheckStrings[StrNo];
    
    StringRef SearchFrom = Buffer;
    
    // Find StrNo in the file.
    size_t MatchLen = 0;
    Buffer = Buffer.substr(CheckStr.Pat.Match(Buffer, MatchLen));
    
    // If we didn't find a match, reject the input.
    if (Buffer.empty()) {
      PrintCheckFailed(SM, CheckStr, SearchFrom);
      return 1;
    }

    StringRef SkippedRegion(LastMatch, Buffer.data()-LastMatch);

    // If this check is a "CHECK-NEXT", verify that the previous match was on
    // the previous line (i.e. that there is one newline between them).
    if (CheckStr.IsCheckNext) {
      // Count the number of newlines between the previous match and this one.
      assert(LastMatch != F->getBufferStart() &&
             "CHECK-NEXT can't be the first check in a file");

      unsigned NumNewLines = CountNumNewlinesBetween(SkippedRegion);
      if (NumNewLines == 0) {
        SM.PrintMessage(CheckStr.Loc,
                    CheckPrefix+"-NEXT: is on the same line as previous match",
                        "error");
        SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()),
                        "'next' match was here", "note");
        SM.PrintMessage(SMLoc::getFromPointer(LastMatch),
                        "previous match was here", "note");
        return 1;
      }
      
      if (NumNewLines != 1) {
        SM.PrintMessage(CheckStr.Loc,
                        CheckPrefix+
                        "-NEXT: is not on the line after the previous match",
                        "error");
        SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()),
                        "'next' match was here", "note");
        SM.PrintMessage(SMLoc::getFromPointer(LastMatch),
                        "previous match was here", "note");
        return 1;
      }
    }
    
    // If this match had "not strings", verify that they don't exist in the
    // skipped region.
    for (unsigned ChunkNo = 0, e = CheckStr.NotStrings.size(); ChunkNo != e; ++ChunkNo) {
      size_t MatchLen = 0;
      size_t Pos = CheckStr.NotStrings[ChunkNo].second.Match(SkippedRegion, MatchLen);
      if (Pos == StringRef::npos) continue;
     
      SM.PrintMessage(SMLoc::getFromPointer(LastMatch+Pos),
                      CheckPrefix+"-NOT: string occurred!", "error");
      SM.PrintMessage(CheckStr.NotStrings[ChunkNo].first,
                      CheckPrefix+"-NOT: pattern specified here", "note");
      return 1;
    }
    

    // Otherwise, everything is good.  Step over the matched text and remember
    // the position after the match as the end of the last match.
    Buffer = Buffer.substr(MatchLen);
    LastMatch = Buffer.data();
  }
  
  return 0;
}
