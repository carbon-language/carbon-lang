//===--- OptTable.cpp - Option Table Implementation ---------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/OptTable.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Option.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <map>
using namespace clang::driver;
using namespace clang::driver::options;

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

namespace clang {
namespace driver {
static inline bool operator<(const OptTable::Info &A, const OptTable::Info &B) {
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

// Support lower_bound between info and an option name.
static inline bool operator<(const OptTable::Info &I, const char *Name) {
  return StrCmpOptionName(I.Name, Name) == -1;
}
static inline bool operator<(const char *Name, const OptTable::Info &I) {
  return StrCmpOptionName(Name, I.Name) == -1;
}
}
}

//

OptSpecifier::OptSpecifier(const Option *Opt) : ID(Opt->getID()) {}

//

OptTable::OptTable(const Info *_OptionInfos, unsigned _NumOptionInfos)
  : OptionInfos(_OptionInfos), NumOptionInfos(_NumOptionInfos),
    Options(new Option*[NumOptionInfos]),
    TheInputOption(0), TheUnknownOption(0), FirstSearchableIndex(0)
{
  // Explicitly zero initialize the error to work around a bug in array
  // value-initialization on MinGW with gcc 4.3.5.
  memset(Options, 0, sizeof(*Options) * NumOptionInfos);

  // Find start of normal options.
  for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
    unsigned Kind = getInfo(i + 1).Kind;
    if (Kind == Option::InputClass) {
      assert(!TheInputOption && "Cannot have multiple input options!");
      TheInputOption = getOption(i + 1);
    } else if (Kind == Option::UnknownClass) {
      assert(!TheUnknownOption && "Cannot have multiple input options!");
      TheUnknownOption = getOption(i + 1);
    } else if (Kind != Option::GroupClass) {
      FirstSearchableIndex = i;
      break;
    }
  }
  assert(FirstSearchableIndex != 0 && "No searchable options?");

#ifndef NDEBUG
  // Check that everything after the first searchable option is a
  // regular option class.
  for (unsigned i = FirstSearchableIndex, e = getNumOptions(); i != e; ++i) {
    Option::OptionClass Kind = (Option::OptionClass) getInfo(i + 1).Kind;
    assert((Kind != Option::InputClass && Kind != Option::UnknownClass &&
            Kind != Option::GroupClass) &&
           "Special options should be defined first!");
  }

  // Check that options are in order.
  for (unsigned i = FirstSearchableIndex+1, e = getNumOptions(); i != e; ++i) {
    if (!(getInfo(i) < getInfo(i + 1))) {
      getOption(i)->dump();
      getOption(i + 1)->dump();
      assert(0 && "Options are not in order!");
    }
  }
#endif
}

OptTable::~OptTable() {
  for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
    delete Options[i];
  delete[] Options;
}

Option *OptTable::CreateOption(unsigned id) const {
  const Info &info = getInfo(id);
  const OptionGroup *Group =
    cast_or_null<OptionGroup>(getOption(info.GroupID));
  const Option *Alias = getOption(info.AliasID);

  Option *Opt = 0;
  switch (info.Kind) {
  case Option::InputClass:
    Opt = new InputOption(id); break;
  case Option::UnknownClass:
    Opt = new UnknownOption(id); break;
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

  if (info.Flags & DriverOption)
    Opt->setDriverOption(true);
  if (info.Flags & LinkerInput)
    Opt->setLinkerInput(true);
  if (info.Flags & NoArgumentUnused)
    Opt->setNoArgumentUnused(true);
  if (info.Flags & NoForward)
    Opt->setNoForward(true);
  if (info.Flags & RenderAsInput)
    Opt->setNoOptAsInput(true);
  if (info.Flags & RenderJoined) {
    assert((info.Kind == Option::JoinedOrSeparateClass ||
            info.Kind == Option::SeparateClass) && "Invalid option.");
    Opt->setRenderStyle(Option::RenderJoinedStyle);
  }
  if (info.Flags & RenderSeparate) {
    assert((info.Kind == Option::JoinedOrSeparateClass ||
            info.Kind == Option::JoinedClass) && "Invalid option.");
    Opt->setRenderStyle(Option::RenderSeparateStyle);
  }
  if (info.Flags & Unsupported)
    Opt->setUnsupported(true);

  return Opt;
}

Arg *OptTable::ParseOneArg(const ArgList &Args, unsigned &Index) const {
  unsigned Prev = Index;
  const char *Str = Args.getArgString(Index);

  // Anything that doesn't start with '-' is an input, as is '-' itself.
  if (Str[0] != '-' || Str[1] == '\0')
    return new Arg(TheInputOption, Index++, Str);

  const Info *Start = OptionInfos + FirstSearchableIndex;
  const Info *End = OptionInfos + getNumOptions();

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
    if (Arg *A = getOption(Start - OptionInfos + 1)->accept(Args, Index))
      return A;

    // Otherwise, see if this argument was missing values.
    if (Prev != Index)
      return 0;
  }

  return new Arg(TheUnknownOption, Index++, Str);
}

InputArgList *OptTable::ParseArgs(const char* const *ArgBegin,
                                  const char* const *ArgEnd,
                                  unsigned &MissingArgIndex,
                                  unsigned &MissingArgCount) const {
  InputArgList *Args = new InputArgList(ArgBegin, ArgEnd);

  // FIXME: Handle '@' args (or at least error on them).

  MissingArgIndex = MissingArgCount = 0;
  unsigned Index = 0, End = ArgEnd - ArgBegin;
  while (Index < End) {
    // Ignore empty arguments (other things may still take them as arguments).
    if (Args->getArgString(Index)[0] == '\0') {
      ++Index;
      continue;
    }

    unsigned Prev = Index;
    Arg *A = ParseOneArg(*Args, Index);
    assert(Index > Prev && "Parser failed to consume argument.");

    // Check for missing argument error.
    if (!A) {
      assert(Index >= End && "Unexpected parser error.");
      assert(Index - Prev - 1 && "No missing arguments!");
      MissingArgIndex = Prev;
      MissingArgCount = Index - Prev - 1;
      break;
    }

    Args->append(A);
  }

  return Args;
}

static std::string getOptionHelpName(const OptTable &Opts, OptSpecifier Id) {
  std::string Name = Opts.getOptionName(Id);

  // Add metavar, if used.
  switch (Opts.getOptionKind(Id)) {
  case Option::GroupClass: case Option::InputClass: case Option::UnknownClass:
    assert(0 && "Invalid option with help text.");

  case Option::MultiArgClass:
    assert(0 && "Cannot print metavar for this kind of option.");

  case Option::FlagClass:
    break;

  case Option::SeparateClass: case Option::JoinedOrSeparateClass:
    Name += ' ';
    // FALLTHROUGH
  case Option::JoinedClass: case Option::CommaJoinedClass:
  case Option::JoinedAndSeparateClass:
    if (const char *MetaVarName = Opts.getOptionMetaVar(Id))
      Name += MetaVarName;
    else
      Name += "<value>";
    break;
  }

  return Name;
}

static void PrintHelpOptionList(llvm::raw_ostream &OS, llvm::StringRef Title,
                                std::vector<std::pair<std::string,
                                const char*> > &OptionHelp) {
  OS << Title << ":\n";

  // Find the maximum option length.
  unsigned OptionFieldWidth = 0;
  for (unsigned i = 0, e = OptionHelp.size(); i != e; ++i) {
    // Skip titles.
    if (!OptionHelp[i].second)
      continue;

    // Limit the amount of padding we are willing to give up for alignment.
    unsigned Length = OptionHelp[i].first.size();
    if (Length <= 23)
      OptionFieldWidth = std::max(OptionFieldWidth, Length);
  }

  const unsigned InitialPad = 2;
  for (unsigned i = 0, e = OptionHelp.size(); i != e; ++i) {
    const std::string &Option = OptionHelp[i].first;
    int Pad = OptionFieldWidth - int(Option.size());
    OS.indent(InitialPad) << Option;

    // Break on long option names.
    if (Pad < 0) {
      OS << "\n";
      Pad = OptionFieldWidth + InitialPad;
    }
    OS.indent(Pad + 1) << OptionHelp[i].second << '\n';
  }
}

static const char *getOptionHelpGroup(const OptTable &Opts, OptSpecifier Id) {
  unsigned GroupID = Opts.getOptionGroupID(Id);

  // If not in a group, return the default help group.
  if (!GroupID)
    return "OPTIONS";

  // Abuse the help text of the option groups to store the "help group"
  // name.
  //
  // FIXME: Split out option groups.
  if (const char *GroupHelp = Opts.getOptionHelpText(GroupID))
    return GroupHelp;

  // Otherwise keep looking.
  return getOptionHelpGroup(Opts, GroupID);
}

void OptTable::PrintHelp(llvm::raw_ostream &OS, const char *Name,
                         const char *Title, bool ShowHidden) const {
  OS << "OVERVIEW: " << Title << "\n";
  OS << '\n';
  OS << "USAGE: " << Name << " [options] <inputs>\n";
  OS << '\n';

  // Render help text into a map of group-name to a list of (option, help)
  // pairs.
  typedef std::map<std::string,
                 std::vector<std::pair<std::string, const char*> > > helpmap_ty;
  helpmap_ty GroupedOptionHelp;

  for (unsigned i = 0, e = getNumOptions(); i != e; ++i) {
    unsigned Id = i + 1;

    // FIXME: Split out option groups.
    if (getOptionKind(Id) == Option::GroupClass)
      continue;

    if (!ShowHidden && isOptionHelpHidden(Id))
      continue;

    if (const char *Text = getOptionHelpText(Id)) {
      const char *HelpGroup = getOptionHelpGroup(*this, Id);
      const std::string &OptName = getOptionHelpName(*this, Id);
      GroupedOptionHelp[HelpGroup].push_back(std::make_pair(OptName, Text));
    }
  }

  for (helpmap_ty::iterator it = GroupedOptionHelp .begin(),
         ie = GroupedOptionHelp.end(); it != ie; ++it) {
    if (it != GroupedOptionHelp .begin())
      OS << "\n";
    PrintHelpOptionList(OS, it->first, it->second);
  }

  OS.flush();
}
