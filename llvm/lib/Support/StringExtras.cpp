//===-- StringExtras.cpp - Implement the StringExtras header --------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the StringExtras.h header
//
//===----------------------------------------------------------------------===//

#include "Support/StringExtras.h"
using namespace llvm;

/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The Source source string is updated in place to remove the returned string
/// and any delimiter prefix from it.
std::string llvm::getToken(std::string &Source, const char *Delimiters) {
  unsigned NumDelimiters = std::strlen(Delimiters);

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
