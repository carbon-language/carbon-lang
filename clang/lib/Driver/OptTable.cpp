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
#include <cassert>

using namespace clang::driver;
using namespace clang::driver::options;

struct Info {
  const char *Name;
  const char *Flags;

  Option::OptionClass Kind;
  unsigned GroupID;
  unsigned AliasID;
  unsigned Param;
};

static Info OptionInfos[] = {
  // The InputOption info
  { "<input>", "", Option::InputClass, OPT_INVALID, OPT_INVALID, 0 },
  // The UnknownOption info
  { "<unknown>", "", Option::UnknownClass, OPT_INVALID, OPT_INVALID, 0 },
  
#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM)              \
  { NAME, FLAGS, Option::KIND##Class, OPT_##GROUP, OPT_##ALIAS, PARAM },
#include "clang/Driver/Options.def"
};
static const unsigned numOptions = sizeof(OptionInfos) / sizeof(OptionInfos[0]);

static Info &getInfo(unsigned id) {
  assert(id > 0 && id - 1 < numOptions && "Invalid Option ID.");
  return OptionInfos[id - 1];
}

OptTable::OptTable() : Options(new Option*[numOptions]()) { 
}

OptTable::~OptTable() { 
  for (unsigned i=0; i<numOptions; ++i)
    delete Options[i];
  delete[] Options;
}

unsigned OptTable::getNumOptions() const {
  return numOptions;
}

const char *OptTable::getOptionName(options::ID id) const {
  return getInfo(id).Name;
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
    case 'J': Opt->setForceJoinedRender(true); break;
    case 'S': Opt->setForceSeparateRender(true); break;
    case 'i': Opt->setNoOptAsInput(true); break;
    case 'l': Opt->setLinkerInput(true); break;
    case 'u': Opt->setUnsupported(true); break;
    }
  }

  return Opt;
}

Arg *OptTable::ParseOneArg(const ArgList &Args, unsigned &Index, 
                           unsigned IndexEnd) const {
  const char *Str = Args.getArgString(Index);

  // Anything that doesn't start with '-' is an input, as is '-' itself.
  if (Str[0] != '-' || Str[1] == '\0')
    return new PositionalArg(getOption(OPT_INPUT), Index++);

  for (unsigned j = OPT_UNKNOWN + 1; j < LastOption; ++j) {
    const char *OptName = getOptionName((options::ID) j);
    
    // Arguments are only accepted by options which prefix them.
    if (memcmp(Str, OptName, strlen(OptName)) == 0)
      if (Arg *A = getOption((options::ID) j)->accept(Args, Index))
        return A;
  }

  return new PositionalArg(getOption(OPT_UNKNOWN), Index++);
}

