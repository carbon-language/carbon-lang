//===-- Regex.cpp - Regular Expression matcher implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a POSIX regular expression matcher.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Regex.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "regex_impl.h"
#include <string>
using namespace llvm;

Regex::Regex(const StringRef &regex, unsigned Flags) {
  unsigned flags = 0;
  preg = new llvm_regex();
  preg->re_endp = regex.end();
  if (Flags & IgnoreCase) 
    flags |= REG_ICASE;
  if (Flags & NoSub) {
    flags |= REG_NOSUB;
    sub = false;
  } else {
    sub = true;
  }
  if (Flags & Newline)
    flags |= REG_NEWLINE;
  error = llvm_regcomp(preg, regex.data(), flags|REG_EXTENDED|REG_PEND);
}

bool Regex::isValid(std::string &Error) {
  if (!error)
    return true;

  size_t len = llvm_regerror(error, preg, NULL, 0);
  
  Error.resize(len);
  llvm_regerror(error, preg, &Error[0], len);
  return false;
}

Regex::~Regex() {
  llvm_regfree(preg);
  delete preg;
}

bool Regex::match(const StringRef &String, SmallVectorImpl<StringRef> *Matches){
  unsigned nmatch = Matches ? preg->re_nsub+1 : 0;

  if (Matches) {
    assert(sub && "Substring matching requested but pattern compiled without");
    Matches->clear();
  }

  // pmatch needs to have at least one element.
  SmallVector<llvm_regmatch_t, 8> pm;
  pm.resize(nmatch > 0 ? nmatch : 1);
  pm[0].rm_so = 0;
  pm[0].rm_eo = String.size();

  int rc = llvm_regexec(preg, String.data(), nmatch, pm.data(), REG_STARTEND);

  if (rc == REG_NOMATCH)
    return false;
  if (rc != 0) {
    // regexec can fail due to invalid pattern or running out of memory.
    error = rc;
    return false;
  }

  // There was a match.

  if (Matches) { // match position requested
    for (unsigned i = 0; i != nmatch; ++i) {
      if (pm[i].rm_so == -1) {
        // this group didn't match
        Matches->push_back(StringRef());
        continue;
      }
      assert(pm[i].rm_eo > pm[i].rm_so);
      Matches->push_back(StringRef(String.data()+pm[i].rm_so,
                                   pm[i].rm_eo-pm[i].rm_so));
    }
  }

  return true;
}
