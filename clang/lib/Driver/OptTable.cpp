//===--- Options.cpp - Option info table --------------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Options.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Option.h"
#include <algorithm>
#include <cassert>

using namespace clang::driver;
using namespace clang::driver::options;

struct Info {
  const char *Name;
  const char *Flags;
  const char *HelpText;
  const char *MetaVar;

  Option::OptionClass Kind;
  unsigned GroupID;
  unsigned AliasID;
  unsigned Param;
};

// Ordering on Info. The ordering is *almost* lexicographic, with two
// exceptions. First, '\0' comes at the end of the alphabet instead of
// the beginning (thus options preceed any other options which prefix
// them). Second, for options with the same name, the less permissive
// version should come first; a Flag option should preceed a Joined
// option, for example.

static int StrCmpOptionName(const char *A, const char *B) {
  char a = *A, b = *B;
  while (a == b) {
    if (a == '\0')
      return 0;

    a = *++A;
    b = *++B;
  }

  if (a == '\0') // A is a prefix of B.
    return 1;
  if (b == '\0') // B is a prefix of A.
    return -1;

  // Otherwise lexicographic.
  return (a < b) ? -1 : 1;
}

static inline bool operator<(const Info &A, const Info &B) {
  if (&A == &B)
    return false;

  if (int N = StrCmpOptionName(A.Name, B.Name))
    return N == -1;

  // Names are the same, check that classes are in order; exactly one
  // should be joined, and it should succeed the other.
  assert(((A.Kind == Option::JoinedClass) ^ (B.Kind == Option::JoinedClass)) &&
         "Unexpected classes for options with same name.");
  return B.Kind == Option::JoinedClass;
}

//

static Info OptionInfos[] = {
  // The InputOption info
  { "<input>", "d", 0, 0, Option::InputClass, OPT_INVALID, OPT_INVALID, 0 },
  // The UnknownOption info
  { "<unknown>", "", 0, 0, Option::UnknownClass, OPT_INVALID, OPT_INVALID, 0 },

#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { NAME, FLAGS, HELPTEXT, METAVAR, \
    Option::KIND##Class, OPT_##GROUP, OPT_##ALIAS, PARAM },
#include "clang/Driver/Options.def"
};
static const unsigned numOptions = sizeof(OptionInfos) / sizeof(OptionInfos[0]);

static Info &getInfo(unsigned id) {
  assert(id > 0 && id - 1 < numOptions && "Invalid Option ID.");
  return OptionInfos[id - 1];
}

OptTable::OptTable() : Options(new Option*[numOptions]) {
  // Explicitly zero initialize the error to work around a bug in array
  // value-initialization on MinGW with gcc 4.3.5.
  memset(Options, 0, sizeof(*Options) * numOptions);

  // Find start of normal options.
  FirstSearchableOption = 0;
  for (unsigned i = OPT_UNKNOWN + 1; i < LastOption; ++i) {
    if (getInfo(i).Kind != Option::GroupClass) {
      FirstSearchableOption = i;
      break;
    }
  }
  assert(FirstSearchableOption != 0 && "No searchable options?");

#ifndef NDEBUG
  // Check that everything after the first searchable option is a
  // regular option class.
  for (unsigned i = FirstSearchableOption; i < LastOption; ++i) {
    Option::OptionClass Kind = getInfo(i).Kind;
    assert((Kind != Option::InputClass && Kind != Option::UnknownClass &&
            Kind != Option::GroupClass) &&
           "Special options should be defined first!");
  }

  // Check that options are in order.
  for (unsigned i = FirstSearchableOption + 1; i < LastOption; ++i) {
    if (!(getInfo(i - 1) < getInfo(i))) {
      getOption((options::ID) (i - 1))->dump();
      getOption((options::ID) i)->dump();
      assert(0 && "Options are not in order!");
    }
  }
#endif
}

OptTable::~OptTable() {
  for (unsigned i = 0; i < numOptions; ++i)
    delete Options[i];
  delete[] Options;
}

unsigned OptTable::getNumOptions() const {
  return numOptions;
}

const char *OptTable::getOptionName(options::ID id) const {
  return getInfo(id).Name;
}

unsigned OptTable::getOptionKind(options::ID id) const {
  return getInfo(id).Kind;
}

const char *OptTable::getOptionHelpText(options::ID id) const {
  return getInfo(id).HelpText;
}

const char *OptTable::getOptionMetaVar(options::ID id) const {
  return getInfo(id).MetaVar;
}

const Option *OptTable::getOption(options::ID id) const {
  if (id == OPT_INVALID)
    return 0;

  assert((unsigned) (id - 1) < numOptions && "Invalid ID.");

  Option *&Entry = Options[id - 1];
  if (!Entry)
    Entry = constructOption(id);

  return Entry;
}

Option *OptTable::constructOption(options::ID id) const {
  Info &info = getInfo(id);
  const OptionGroup *Group =
    cast_or_null<OptionGroup>(getOption((options::ID) info.GroupID));
  const Option *Alias = getOption((options::ID) info.AliasID);

  Option *Opt = 0;
  switch (info.Kind) {
  case Option::InputClass:
    Opt = new InputOption(); break;
  case Option::UnknownClass:
    Opt = new UnknownOption(); break;
  case Option::GroupClass:
    Opt = new OptionGroup(id, info.Name, Group); break;
  case Option::FlagClass:
    Opt = new FlagOption(id, info.Name, Group, Alias); break;
  case Option::JoinedClass:
    Opt = new JoinedOption(id, info.Name, Group, Alias); break;
  case Option::SeparateClass:
    Opt = new SeparateOption(id, info.Name, Group, Alias); break;
  case Option::CommaJoinedClass:
    Opt = new CommaJoinedOption(id, info.Name, Group, Alias); break;
  case Option::MultiArgClass:
    Opt = new MultiArgOption(id, info.Name, Group, Alias, info.Param); break;
  case Option::JoinedOrSeparateClass:
    Opt = new JoinedOrSeparateOption(id, info.Name, Group, Alias); break;
  case Option::JoinedAndSeparateClass:
    Opt = new JoinedAndSeparateOption(id, info.Name, Group, Alias); break;
  }

  for (const char *s = info.Flags; *s; ++s) {
    switch (*s) {
    default: assert(0 && "Invalid option flag.");
    case 'J':
      assert(info.Kind == Option::SeparateClass && "Invalid option.");
      Opt->setForceJoinedRender(true); break;
    case 'S':
      assert(info.Kind == Option::JoinedClass && "Invalid option.");
      Opt->setForceSeparateRender(true); break;
    case 'd': Opt->setDriverOption(true); break;
    case 'i': Opt->setNoOptAsInput(true); break;
    case 'l': Opt->setLinkerInput(true); break;
    case 'q': Opt->setNoArgumentUnused(true); break;
    case 'u': Opt->setUnsupported(true); break;
    }
  }

  return Opt;
}

// Support lower_bound between info and an option name.
static inline bool operator<(struct Info &I, const char *Name) {
  return StrCmpOptionName(I.Name, Name) == -1;
}
static inline bool operator<(const char *Name, struct Info &I) {
  return StrCmpOptionName(Name, I.Name) == -1;
}

Arg *OptTable::ParseOneArg(const InputArgList &Args, unsigned &Index) const {
  unsigned Prev = Index;
  const char *Str = Args.getArgString(Index);

  // Anything that doesn't start with '-' is an input, as is '-' itself.
  if (Str[0] != '-' || Str[1] == '\0')
    return new PositionalArg(getOption(OPT_INPUT), Index++);

  struct Info *Start = OptionInfos + FirstSearchableOption - 1;
  struct Info *End = OptionInfos + LastOption - 1;

  // Search for the first next option which could be a prefix.
  Start = std::lower_bound(Start, End, Str);

  // Options are stored in sorted order, with '\0' at the end of the
  // alphabet. Since the only options which can accept a string must
  // prefix it, we iteratively search for the next option which could
  // be a prefix.
  //
  // FIXME: This is searching much more than necessary, but I am
  // blanking on the simplest way to make it fast. We can solve this
  // problem when we move to TableGen.
  for (; Start != End; ++Start) {
    // Scan for first option which is a proper prefix.
    for (; Start != End; ++Start)
      if (memcmp(Str, Start->Name, strlen(Start->Name)) == 0)
        break;
    if (Start == End)
      break;

    // See if this option matches.
    options::ID id = (options::ID) (Start - OptionInfos + 1);
    if (Arg *A = getOption(id)->accept(Args, Index))
      return A;

    // Otherwise, see if this argument was missing values.
    if (Prev != Index)
      return 0;
  }

  return new PositionalArg(getOption(OPT_UNKNOWN), Index++);
}

