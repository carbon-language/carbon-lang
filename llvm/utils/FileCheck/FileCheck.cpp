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

/// CheckString - This is a check that we found in the input file.
struct CheckString {
  /// Str - The string to match.
  std::string Str;
  
  /// Loc - The location in the match file that the check string was specified.
  SMLoc Loc;
  
  /// IsCheckNext - This is true if this is a CHECK-NEXT: directive (as opposed
  /// to a CHECK: directive.
  bool IsCheckNext;
  
  /// NotStrings - These are all of the strings that are disallowed from
  /// occurring between this match string and the previous one (or start of
  /// file).
  std::vector<std::pair<SMLoc, std::string> > NotStrings;
  
  CheckString(const std::string &S, SMLoc L, bool isCheckNext)
    : Str(S), Loc(L), IsCheckNext(isCheckNext) {}
};


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
  SM.AddNewSourceBuffer(F, SMLoc());

  // Find all instances of CheckPrefix followed by : in the file.
  StringRef Buffer = F->getBuffer();

  std::vector<std::pair<SMLoc, std::string> > NotMatches;
  
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
    if (EOL == StringRef::npos) EOL = Buffer.size();
    
    // Ignore trailing whitespace.
    while (EOL && (Buffer[EOL-1] == ' ' || Buffer[EOL-1] == '\t'))
      --EOL;
    
    // Check that there is something on the line.
    if (EOL == 0) {
      SM.PrintMessage(SMLoc::getFromPointer(Buffer.data()),
                      "found empty check string with prefix '"+CheckPrefix+":'",
                      "error");
      return true;
    }
    
    StringRef PatternStr = Buffer.substr(0, EOL);
    
    // Handle CHECK-NOT.
    if (IsCheckNot) {
      NotMatches.push_back(std::make_pair(SMLoc::getFromPointer(Buffer.data()),
                                          PatternStr.str()));
      Buffer = Buffer.substr(EOL);
      continue;
    }
    
    // Verify that CHECK-NEXT lines have at least one CHECK line before them.
    if (IsCheckNext && CheckStrings.empty()) {
      SM.PrintMessage(SMLoc::getFromPointer(CheckPrefixStart),
                      "found '"+CheckPrefix+"-NEXT:' without previous '"+
                      CheckPrefix+ ": line", "error");
      return true;
    }
    
    // Okay, add the string we captured to the output vector and move on.
    CheckStrings.push_back(CheckString(PatternStr.str(),
                                       SMLoc::getFromPointer(Buffer.data()),
                                       IsCheckNext));
    std::swap(NotMatches, CheckStrings.back().NotStrings);
    
    Buffer = Buffer.substr(EOL);
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

// CanonicalizeCheckStrings - Replace all sequences of horizontal whitespace in
// the check strings with a single space.
static void CanonicalizeCheckStrings(std::vector<CheckString> &CheckStrings) {
  for (unsigned i = 0, e = CheckStrings.size(); i != e; ++i) {
    std::string &Str = CheckStrings[i].Str;
    
    for (unsigned C = 0; C != Str.size(); ++C) {
      // If C is not a horizontal whitespace, skip it.
      if (Str[C] != ' ' && Str[C] != '\t')
        continue;
      
      // Replace the character with space, then remove any other space
      // characters after it.
      Str[C] = ' ';
      
      while (C+1 != Str.size() &&
             (Str[C+1] == ' ' || Str[C+1] == '\t'))
        Str.erase(Str.begin()+C+1);
    }
  }
}

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

static unsigned CountNumNewlinesBetween(const char *Start, const char *End) {
  unsigned NumNewLines = 0;
  for (; Start != End; ++Start) {
    // Scan for newline.
    if (Start[0] != '\n' && Start[0] != '\r')
      continue;
    
    ++NumNewLines;
    
    // Handle \n\r and \r\n as a single newline.
    if (Start+1 != End &&
        (Start[0] == '\n' || Start[0] == '\r') &&
        (Start[0] != Start[1]))
      ++Start;
  }
  
  return NumNewLines;
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

  // Remove duplicate spaces in the check strings if requested.
  if (!NoCanonicalizeWhiteSpace)
    CanonicalizeCheckStrings(CheckStrings);

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
    Buffer = Buffer.substr(Buffer.find(CheckStr.Str));
    
    // If we didn't find a match, reject the input.
    if (Buffer.empty()) {
      PrintCheckFailed(SM, CheckStr, SearchFrom);
      return 1;
    }
    
    // If this check is a "CHECK-NEXT", verify that the previous match was on
    // the previous line (i.e. that there is one newline between them).
    if (CheckStr.IsCheckNext) {
      // Count the number of newlines between the previous match and this one.
      assert(LastMatch != F->getBufferStart() &&
             "CHECK-NEXT can't be the first check in a file");

      unsigned NumNewLines = CountNumNewlinesBetween(LastMatch, Buffer.data());
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
    StringRef SkippedRegion(LastMatch, Buffer.data()-LastMatch);
    for (unsigned i = 0, e = CheckStr.NotStrings.size(); i != e; ++i) {
      size_t Pos = SkippedRegion.find(CheckStr.NotStrings[i].second);
      if (Pos == StringRef::npos) continue;
     
      SM.PrintMessage(SMLoc::getFromPointer(LastMatch+Pos),
                      CheckPrefix+"-NOT: string occurred!", "error");
      SM.PrintMessage(CheckStr.NotStrings[i].first,
                      CheckPrefix+"-NOT: pattern specified here", "note");
      return 1;
    }
    

    // Otherwise, everything is good.  Remember this as the last match and move
    // on to the next one.
    LastMatch = Buffer.data();
    Buffer = Buffer.substr(CheckStr.Str.size());
  }
  
  return 0;
}
