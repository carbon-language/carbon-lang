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
  
  CheckString(const std::string &S, SMLoc L, bool isCheckNext)
    : Str(S), Loc(L), IsCheckNext(isCheckNext) {}
};


/// FindFixedStringInBuffer - This works like strstr, except for two things:
/// 1) it handles 'nul' characters in memory buffers.  2) it returns the end of
/// the memory buffer on match failure instead of null.
static const char *FindFixedStringInBuffer(StringRef Str, const char *CurPtr,
                                           const MemoryBuffer &MB) {
  assert(!Str.empty() && "Can't find an empty string");
  const char *BufEnd = MB.getBufferEnd();
  
  while (1) {
    // Scan for the first character in the match string.
    CurPtr = (char*)memchr(CurPtr, Str[0], BufEnd-CurPtr);
    
    // If we didn't find the first character of the string, then we failed to
    // match.
    if (CurPtr == 0) return BufEnd;

    // If the match string is one character, then we win.
    if (Str.size() == 1) return CurPtr;
    
    // Otherwise, verify that the rest of the string matches.
    if (Str.size() <= unsigned(BufEnd-CurPtr) &&
        memcmp(CurPtr+1, Str.data()+1, Str.size()-1) == 0)
      return CurPtr;
    
    // If not, advance past this character and try again.
    ++CurPtr;
  }
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
  SM.AddNewSourceBuffer(F, SMLoc());

  // Find all instances of CheckPrefix followed by : in the file.
  const char *CurPtr = F->getBufferStart(), *BufferEnd = F->getBufferEnd();

  while (1) {
    // See if Prefix occurs in the memory buffer.
    const char *Ptr = FindFixedStringInBuffer(CheckPrefix, CurPtr, *F);
    
    // If we didn't find a match, we're done.
    if (Ptr == BufferEnd)
      break;
    
    const char *CheckPrefixStart = Ptr;
    
    // When we find a check prefix, keep track of whether we find CHECK: or
    // CHECK-NEXT:
    bool IsCheckNext;
    
    // Verify that the : is present after the prefix.
    if (Ptr[CheckPrefix.size()] == ':') {
      Ptr += CheckPrefix.size()+1;
      IsCheckNext = false;
    } else if (BufferEnd-Ptr > 6 &&
               memcmp(Ptr+CheckPrefix.size(), "-NEXT:", 6) == 0) {
      Ptr += CheckPrefix.size()+7;
      IsCheckNext = true;
    } else {
      CurPtr = Ptr+1;
      continue;
    }
    
    // Okay, we found the prefix, yay.  Remember the rest of the line, but
    // ignore leading and trailing whitespace.
    while (*Ptr == ' ' || *Ptr == '\t')
      ++Ptr;
    
    // Scan ahead to the end of line.
    CurPtr = Ptr;
    while (CurPtr != BufferEnd && *CurPtr != '\n' && *CurPtr != '\r')
      ++CurPtr;
    
    // Ignore trailing whitespace.
    while (CurPtr[-1] == ' ' || CurPtr[-1] == '\t')
      --CurPtr;
    
    // Check that there is something on the line.
    if (Ptr >= CurPtr) {
      SM.PrintMessage(SMLoc::getFromPointer(CurPtr),
                      "found empty check string with prefix '"+CheckPrefix+":'",
                      "error");
      return true;
    }
    
    // Verify that CHECK-NEXT lines have at least one CHECK line before them.
    if (IsCheckNext && CheckStrings.empty()) {
      SM.PrintMessage(SMLoc::getFromPointer(CheckPrefixStart),
                      "found '"+CheckPrefix+"-NEXT:' without previous '"+
                      CheckPrefix+ ": line", "error");
      return true;
    }
    
    // Okay, add the string we captured to the output vector and move on.
    CheckStrings.push_back(CheckString(std::string(Ptr, CurPtr),
                                       SMLoc::getFromPointer(Ptr),
                                       IsCheckNext));
  }
  
  if (CheckStrings.empty()) {
    errs() << "error: no check strings found with prefix '" << CheckPrefix
           << ":'\n";
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
                             const char *CurPtr, const char *BufferEnd) {
  // Otherwise, we have an error, emit an error message.
  SM.PrintMessage(CheckStr.Loc, "expected string not found in input",
                  "error");
  
  // Print the "scanning from here" line.  If the current position is at the
  // end of a line, advance to the start of the next line.
  const char *Scan = CurPtr;
  while (Scan != BufferEnd &&
         (*Scan == ' ' || *Scan == '\t'))
    ++Scan;
  if (*Scan == '\n' || *Scan == '\r')
    CurPtr = Scan+1;
  
  
  SM.PrintMessage(SMLoc::getFromPointer(CurPtr), "scanning from here",
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
  const char *CurPtr = F->getBufferStart(), *BufferEnd = F->getBufferEnd();
  
  const char *LastMatch = 0;
  for (unsigned StrNo = 0, e = CheckStrings.size(); StrNo != e; ++StrNo) {
    const CheckString &CheckStr = CheckStrings[StrNo];
    
    // Find StrNo in the file.
    const char *Ptr = FindFixedStringInBuffer(CheckStr.Str, CurPtr, *F);
    
    // If we didn't find a match, reject the input.
    if (Ptr == BufferEnd) {
      PrintCheckFailed(SM, CheckStr, CurPtr, BufferEnd);
      return 1;
    }
    
    // If this check is a "CHECK-NEXT", verify that the previous match was on
    // the previous line (i.e. that there is one newline between them).
    if (CheckStr.IsCheckNext) {
      // Count the number of newlines between the previous match and this one.
      assert(LastMatch && "CHECK-NEXT can't be the first check in a file");

      unsigned NumNewLines = CountNumNewlinesBetween(LastMatch, Ptr);
      if (NumNewLines == 0) {
        SM.PrintMessage(SMLoc::getFromPointer(Ptr),
                    CheckPrefix+"-NEXT: is on the same line as previous match",
                        "error");
        SM.PrintMessage(SMLoc::getFromPointer(LastMatch),
                        "previous match was here", "note");
        return 1;
      }
      
      if (NumNewLines != 1) {
        SM.PrintMessage(SMLoc::getFromPointer(Ptr),
                        CheckPrefix+
                        "-NEXT: is not on the line after the previous match",
                        "error");
        SM.PrintMessage(SMLoc::getFromPointer(LastMatch),
                        "previous match was here", "note");
        return 1;
      }
    }

    // Otherwise, everything is good.  Remember this as the last match and move
    // on to the next one.
    LastMatch = Ptr;
    CurPtr = Ptr + CheckStr.Str.size();
  }
  
  return 0;
}
