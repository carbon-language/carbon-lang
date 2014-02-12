//===--- Multilib.cpp - Multilib Implementation ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Multilib.h"
#include "Tools.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include <algorithm>

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;
using namespace llvm::sys;

/// normalize Segment to "/foo/bar" or "".
static void normalizePathSegment(std::string &Segment) {
  StringRef seg = Segment;

  // Prune trailing "/" or "./"
  while (1) {
    StringRef last = *--path::end(seg);
    if (last != ".")
      break;
    seg = path::parent_path(seg);
  }

  if (seg.empty() || seg == "/") {
    Segment = "";
    return;
  }

  // Add leading '/'
  if (seg.front() != '/') {
    Segment = "/" + seg.str();
  } else {
    Segment = seg;
  }
}

Multilib::Multilib(StringRef GCCSuffix, StringRef OSSuffix,
                   StringRef IncludeSuffix)
    : GCCSuffix(GCCSuffix), OSSuffix(OSSuffix), IncludeSuffix(IncludeSuffix) {
  normalizePathSegment(this->GCCSuffix);
  normalizePathSegment(this->OSSuffix);
  normalizePathSegment(this->IncludeSuffix);
}

Multilib &Multilib::gccSuffix(StringRef S) {
  GCCSuffix = S;
  normalizePathSegment(GCCSuffix);
  return *this;
}

Multilib &Multilib::osSuffix(StringRef S) {
  OSSuffix = S;
  normalizePathSegment(OSSuffix);
  return *this;
}

Multilib &Multilib::includeSuffix(StringRef S) {
  IncludeSuffix = S;
  normalizePathSegment(IncludeSuffix);
  return *this;
}

void Multilib::print(raw_ostream &OS) const {
  assert(GCCSuffix.empty() || (StringRef(GCCSuffix).front() == '/'));
  if (GCCSuffix.empty())
    OS << ".";
  else {
    OS << StringRef(GCCSuffix).drop_front();
  }
  OS << ";";
  for (flags_list::const_iterator I = Flags.begin(), E = Flags.end(); I != E;
       ++I) {
    if (StringRef(*I).front() == '+')
      OS << "@" << I->substr(1);
  }
}

bool Multilib::isValid() const {
  llvm::StringMap<int> FlagSet;
  for (unsigned I = 0, N = Flags.size(); I != N; ++I) {
    StringRef Flag(Flags[I]);
    llvm::StringMap<int>::iterator SI = FlagSet.find(Flag.substr(1));

    assert(StringRef(Flag).front() == '+' || StringRef(Flag).front() == '-');

    if (SI == FlagSet.end())
      FlagSet[Flag.substr(1)] = I;
    else if (Flags[I] != Flags[SI->getValue()])
      return false;
  }
  return true;
}

bool Multilib::operator==(const Multilib &Other) const {
  // Check whether the flags sets match
  // allowing for the match to be order invariant
  llvm::StringSet<> MyFlags;
  for (flags_list::const_iterator I = Flags.begin(), E = Flags.end(); I != E;
       ++I) {
    MyFlags.insert(*I);
  }
  for (flags_list::const_iterator I = Other.Flags.begin(),
                                  E = Other.Flags.end();
       I != E; ++I) {
    if (MyFlags.find(*I) == MyFlags.end())
      return false;
  }

  if (osSuffix() != Other.osSuffix())
    return false;

  if (gccSuffix() != Other.gccSuffix())
    return false;

  if (includeSuffix() != Other.includeSuffix())
    return false;

  return true;
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const Multilib &M) {
  M.print(OS);
  return OS;
}

MultilibSet &MultilibSet::Maybe(const Multilib &M) {
  Multilib Opposite;
  // Negate any '+' flags
  for (Multilib::flags_list::const_iterator I = M.flags().begin(),
                                            E = M.flags().end();
       I != E; ++I) {
    StringRef Flag(*I);
    if (Flag.front() == '+')
      Opposite.flags().push_back(("-" + Flag.substr(1)).str());
  }
  return Either(M, Opposite);
}

MultilibSet &MultilibSet::Either(const Multilib &M1, const Multilib &M2) {
  std::vector<Multilib> Ms;
  Ms.push_back(M1);
  Ms.push_back(M2);
  return Either(Ms);
}

MultilibSet &MultilibSet::Either(const Multilib &M1, const Multilib &M2,
                                 const Multilib &M3) {
  std::vector<Multilib> Ms;
  Ms.push_back(M1);
  Ms.push_back(M2);
  Ms.push_back(M3);
  return Either(Ms);
}

MultilibSet &MultilibSet::Either(const Multilib &M1, const Multilib &M2,
                                 const Multilib &M3, const Multilib &M4) {
  std::vector<Multilib> Ms;
  Ms.push_back(M1);
  Ms.push_back(M2);
  Ms.push_back(M3);
  Ms.push_back(M4);
  return Either(Ms);
}

MultilibSet &MultilibSet::Either(const Multilib &M1, const Multilib &M2,
                                 const Multilib &M3, const Multilib &M4,
                                 const Multilib &M5) {
  std::vector<Multilib> Ms;
  Ms.push_back(M1);
  Ms.push_back(M2);
  Ms.push_back(M3);
  Ms.push_back(M4);
  Ms.push_back(M5);
  return Either(Ms);
}

static Multilib compose(const Multilib &Base, const Multilib &New) {
  SmallString<128> GCCSuffix;
  llvm::sys::path::append(GCCSuffix, "/", Base.gccSuffix(), New.gccSuffix());
  SmallString<128> OSSuffix;
  llvm::sys::path::append(OSSuffix, "/", Base.osSuffix(), New.osSuffix());
  SmallString<128> IncludeSuffix;
  llvm::sys::path::append(IncludeSuffix, "/", Base.includeSuffix(),
                          New.includeSuffix());

  Multilib Composed(GCCSuffix.str(), OSSuffix.str(), IncludeSuffix.str());

  Multilib::flags_list &Flags = Composed.flags();

  Flags.insert(Flags.end(), Base.flags().begin(), Base.flags().end());
  Flags.insert(Flags.end(), New.flags().begin(), New.flags().end());

  return Composed;
}

MultilibSet &
MultilibSet::Either(const std::vector<Multilib> &MultilibSegments) {
  multilib_list Composed;

  if (Multilibs.empty())
    Multilibs.insert(Multilibs.end(), MultilibSegments.begin(),
                     MultilibSegments.end());
  else {
    for (std::vector<Multilib>::const_iterator NewI = MultilibSegments.begin(),
                                               NewE = MultilibSegments.end();
         NewI != NewE; ++NewI) {
      for (const_iterator BaseI = begin(), BaseE = end(); BaseI != BaseE;
           ++BaseI) {
        Multilib MO = compose(*BaseI, *NewI);
        if (MO.isValid())
          Composed.push_back(MO);
      }
    }

    Multilibs = Composed;
  }

  return *this;
}

MultilibSet &MultilibSet::FilterOut(const MultilibSet::FilterCallback &F) {
  filterInPlace(F, Multilibs);
  return *this;
}

MultilibSet &MultilibSet::FilterOut(std::string Regex) {
  class REFilter : public MultilibSet::FilterCallback {
    mutable llvm::Regex R;

  public:
    REFilter(std::string Regex) : R(Regex) {}
    bool operator()(const Multilib &M) const LLVM_OVERRIDE {
      std::string Error;
      if (!R.isValid(Error)) {
        llvm::errs() << Error;
        assert(false);
        return false;
      }
      return R.match(M.gccSuffix());
    }
  };

  REFilter REF(Regex);
  filterInPlace(REF, Multilibs);
  return *this;
}

void MultilibSet::push_back(const Multilib &M) { Multilibs.push_back(M); }

void MultilibSet::combineWith(const MultilibSet &Other) {
  Multilibs.insert(Multilibs.end(), Other.begin(), Other.end());
}

bool MultilibSet::select(const Multilib::flags_list &Flags, Multilib &M) const {
  class FilterFlagsMismatch : public MultilibSet::FilterCallback {
    llvm::StringMap<bool> FlagSet;

  public:
    FilterFlagsMismatch(const std::vector<std::string> &Flags) {
      // Stuff all of the flags into the FlagSet such that a true mappend
      // indicates the flag was enabled, and a false mappend indicates the
      // flag was disabled
      for (Multilib::flags_list::const_iterator I = Flags.begin(),
                                                E = Flags.end();
           I != E; ++I) {
        FlagSet[StringRef(*I).substr(1)] = isFlagEnabled(*I);
      }
    }
    bool operator()(const Multilib &M) const LLVM_OVERRIDE {
      for (Multilib::flags_list::const_iterator I = M.flags().begin(),
                                                E = M.flags().end();
           I != E; ++I) {
        StringRef Flag(*I);
        llvm::StringMap<bool>::const_iterator SI = FlagSet.find(Flag.substr(1));
        if (SI != FlagSet.end())
          if ((*SI).getValue() != isFlagEnabled(Flag))
            return true;
      }
      return false;
    }
  private:
    bool isFlagEnabled(StringRef Flag) const {
      char Indicator = Flag.front();
      assert(Indicator == '+' || Indicator == '-');
      return Indicator == '+';
    }
  };

  FilterFlagsMismatch FlagsMismatch(Flags);

  multilib_list Filtered = filterCopy(FlagsMismatch, Multilibs);

  if (Filtered.size() == 0) {
    return false;
  } else if (Filtered.size() == 1) {
    M = Filtered[0];
    return true;
  }

  // TODO: pick the "best" multlib when more than one is suitable
  assert(false);

  return false;
}

void MultilibSet::print(raw_ostream &OS) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    OS << *I << "\n";
}

MultilibSet::multilib_list
MultilibSet::filterCopy(const MultilibSet::FilterCallback &F,
                        const multilib_list &Ms) {
  multilib_list Copy(Ms);
  filterInPlace(F, Copy);
  return Copy;
}

namespace {
// Wrapper for FilterCallback to make operator() nonvirtual so it
// can be passed by value to std::remove_if
class FilterWrapper {
  const MultilibSet::FilterCallback &F;
public:
  FilterWrapper(const MultilibSet::FilterCallback &F) : F(F) {}
  bool operator()(const Multilib &M) const { return F(M); }
};
} // end anonymous namespace

void MultilibSet::filterInPlace(const MultilibSet::FilterCallback &F,
                                multilib_list &Ms) {
  Ms.erase(std::remove_if(Ms.begin(), Ms.end(), FilterWrapper(F)), Ms.end());
}

raw_ostream &clang::driver::operator<<(raw_ostream &OS, const MultilibSet &MS) {
  MS.print(OS);
  return OS;
}
