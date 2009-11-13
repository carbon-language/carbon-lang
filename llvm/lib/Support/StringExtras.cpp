//===-- StringExtras.cpp - Implement the StringExtras header --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringExtras.h header
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstring>
using namespace llvm;

/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The Source source string is updated in place to remove the returned string
/// and any delimiter prefix from it.
std::string llvm::getToken(std::string &Source, const char *Delimiters) {
  size_t NumDelimiters = std::strlen(Delimiters);

  // Figure out where the token starts.
  std::string::size_type Start =
    Source.find_first_not_of(Delimiters, 0, NumDelimiters);
  if (Start == std::string::npos) Start = Source.size();

  // Find the next occurance of the delimiter.
  std::string::size_type End =
    Source.find_first_of(Delimiters, Start, NumDelimiters);
  if (End == std::string::npos) End = Source.size();

  // Create the return token.
  std::string Result = std::string(Source.begin()+Start, Source.begin()+End);

  // Erase the token that we read in.
  Source.erase(Source.begin(), Source.begin()+End);

  return Result;
}

/// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
void llvm::SplitString(const std::string &Source, 
                       std::vector<std::string> &OutFragments,
                       const char *Delimiters) {
  std::string S = Source;
  
  std::string S2 = getToken(S, Delimiters);
  while (!S2.empty()) {
    OutFragments.push_back(S2);
    S2 = getToken(S, Delimiters);
  }
}

void llvm::StringRef::split(SmallVectorImpl<StringRef> &A,
                            StringRef Separators, int MaxSplit,
                            bool KeepEmpty) const {
  StringRef rest = *this;

  // rest.data() is used to distinguish cases like "a," that splits into
  // "a" + "" and "a" that splits into "a" + 0.
  for (int splits = 0;
       rest.data() != NULL && (MaxSplit < 0 || splits < MaxSplit);
       ++splits) {
    std::pair<llvm::StringRef, llvm::StringRef> p = rest.split(Separators);

    if (p.first.size() != 0 || KeepEmpty)
      A.push_back(p.first);
    rest = p.second;
  }
  // If we have a tail left, add it.
  if (rest.data() != NULL && (rest.size() != 0 || KeepEmpty))
    A.push_back(rest);
}
