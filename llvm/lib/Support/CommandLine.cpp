//===-- CommandLine.cpp - Command line parser implementation --------------===//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
// Note that rather than trying to figure out what this code does, you could try
// reading the library documentation located in docs/CommandLine.html
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <map>
#include <set>
#include <iostream>

using namespace cl;
using std::map;
using std::pair;
using std::vector;
using std::string;
using std::cerr;

// Return the global command line option vector.  Making it a function scoped
// static ensures that it will be initialized correctly before its first use.
//
static map<string, Option*> &getOpts() {
  static map<string,Option*> CommandLineOptions;
  return CommandLineOptions;
}

static void AddArgument(const string &ArgName, Option *Opt) {
  if (getOpts().find(ArgName) != getOpts().end()) {
    cerr << "CommandLine Error: Argument '" << ArgName
	 << "' defined more than once!\n";
  } else {
    // Add argument to the argument map!
    getOpts().insert(std::make_pair(ArgName, Opt));
  }
}

static const char *ProgramName = 0;
static const char *ProgramOverview = 0;

static inline bool ProvideOption(Option *Handler, const char *ArgName,
                                 const char *Value, int argc, char **argv,
                                 int &i) {
  // Enforce value requirements
  switch (Handler->getValueExpectedFlag()) {
  case ValueRequired:
    if (Value == 0 || *Value == 0) {  // No value specified?
      if (i+1 < argc) {     // Steal the next argument, like for '-o filename'
        Value = argv[++i];
      } else {
        return Handler->error(" requires a value!");
      }
    }
    break;
  case ValueDisallowed:
    if (*Value != 0)
      return Handler->error(" does not allow a value! '" + 
                            string(Value) + "' specified.");
    break;
  case ValueOptional: break;
  default: cerr << "Bad ValueMask flag! CommandLine usage error:" 
                << Handler->getValueExpectedFlag() << "\n"; abort();
  }

  // Run the handler now!
  return Handler->addOccurance(ArgName, Value);
}

// ValueGroupedArgs - Return true if the specified string is valid as a group
// of single letter arguments stuck together like the 'ls -la' case.
//
static inline bool ValidGroupedArgs(string Args) {
  for (unsigned i = 0; i < Args.size(); ++i) {
    map<string, Option*>::iterator I = getOpts().find(string(1, Args[i]));
    if (I == getOpts().end()) return false;   // Make sure option exists

    // Grouped arguments have no value specified, make sure that if this option
    // exists that it can accept no argument.
    //
    switch (I->second->getValueExpectedFlag()) {
    case ValueDisallowed:
    case ValueOptional: break;
    default: return false;
    }
  }

  return true;
}

void cl::ParseCommandLineOptions(int &argc, char **argv,
				 const char *Overview = 0, int Flags = 0) {
  ProgramName = argv[0];  // Save this away safe and snug
  ProgramOverview = Overview;
  bool ErrorParsing = false;

  // Loop over all of the arguments... processing them.
  for (int i = 1; i < argc; ++i) {
    Option *Handler = 0;
    const char *Value = "";
    const char *ArgName = "";
    if (argv[i][0] != '-') {   // Unnamed argument?
      map<string, Option*>::iterator I = getOpts().find("");
      Handler = I != getOpts().end() ? I->second : 0;
      Value = argv[i];
    } else {               // We start with a - or --, eat dashes
      ArgName = argv[i]+1;
      while (*ArgName == '-') ++ArgName;  // Eat leading dashes

      const char *ArgNameEnd = ArgName;
      while (*ArgNameEnd && *ArgNameEnd != '=')
	++ArgNameEnd; // Scan till end of argument name...

      Value = ArgNameEnd;
      if (*Value)           // If we have an equals sign...
	++Value;            // Advance to value...

      if (*ArgName != 0) {
	string RealName(ArgName, ArgNameEnd);
	// Extract arg name part
        map<string, Option*>::iterator I = getOpts().find(RealName);

	if (I == getOpts().end() && !*Value && RealName.size() > 1) {
	  // If grouping of single letter arguments is enabled, see if this is a
	  // legal grouping...
	  //
	  if (!(Flags & DisableSingleLetterArgGrouping) &&
	      ValidGroupedArgs(RealName)) {

	    for (unsigned i = 0; i < RealName.size(); ++i) {
	      char ArgName[2] = { 0, 0 }; int Dummy;
	      ArgName[0] = RealName[i];
	      I = getOpts().find(ArgName);
	      assert(I != getOpts().end() && "ValidGroupedArgs failed!");

	      // Because ValueRequired is an invalid flag for grouped arguments,
	      // we don't need to pass argc/argv in...
	      //
	      ErrorParsing |= ProvideOption(I->second, ArgName, "",
					    0, 0, Dummy);
	    }
	    continue;
	  } else if (Flags & EnableSingleLetterArgValue) {
	    // Check to see if the first letter is a single letter argument that
	    // have a value that is equal to the rest of the string.  If this
	    // is the case, recognize it now.  (Example:  -lfoo for a linker)
	    //
	    I = getOpts().find(string(1, RealName[0]));
	    if (I != getOpts().end()) {
	      // If we are successful, fall through to later processing, by
	      // setting up the argument name flags and value fields.
	      //
	      ArgNameEnd = ArgName+1;
	      Value = ArgNameEnd;
	    }
	  }
	}


        Handler = I != getOpts().end() ? I->second : 0;
      }
    }

    if (Handler == 0) {
      cerr << "Unknown command line argument '" << argv[i] << "'.  Try: "
	   << argv[0] << " --help'\n";
      ErrorParsing = true;
      continue;
    }

    ErrorParsing |= ProvideOption(Handler, ArgName, Value, argc, argv, i);

    // If this option should consume all arguments that come after it...
    if (Handler->getNumOccurancesFlag() == ConsumeAfter) {
      for (++i; i < argc; ++i)
        ErrorParsing |= ProvideOption(Handler, ArgName, argv[i], argc, argv, i);
    }
  }

  // Loop over args and make sure all required args are specified!
  for (map<string, Option*>::iterator I = getOpts().begin(), 
	 E = getOpts().end(); I != E; ++I) {
    switch (I->second->getNumOccurancesFlag()) {
    case Required:
    case OneOrMore:
      if (I->second->getNumOccurances() == 0) {
	I->second->error(" must be specified at least once!");
        ErrorParsing = true;
      }
      // Fall through
    default:
      break;
    }
  }

  // Free all of the memory allocated to the vector.  Command line options may
  // only be processed once!
  getOpts().clear();

  // If we had an error processing our arguments, don't let the program execute
  if (ErrorParsing) exit(1);
}

//===----------------------------------------------------------------------===//
// Option Base class implementation
//
Option::Option(const char *argStr, const char *helpStr, int flags)
  : NumOccurances(0), Flags(flags), ArgStr(argStr), HelpStr(helpStr) {
  AddArgument(ArgStr, this);
}

bool Option::error(string Message, const char *ArgName = 0) {
  if (ArgName == 0) ArgName = ArgStr;
  cerr << "-" << ArgName << " option" << Message << "\n";
  return true;
}

bool Option::addOccurance(const char *ArgName, const string &Value) {
  NumOccurances++;   // Increment the number of times we have been seen

  switch (getNumOccurancesFlag()) {
  case Optional:
    if (NumOccurances > 1)
      return error(": may only occur zero or one times!", ArgName);
    break;
  case Required:
    if (NumOccurances > 1)
      return error(": must occur exactly one time!", ArgName);
    // Fall through
  case OneOrMore:
  case ZeroOrMore:
  case ConsumeAfter: break;
  default: return error(": bad num occurances flag value!");
  }

  return handleOccurance(ArgName, Value);
}

// Return the width of the option tag for printing...
unsigned Option::getOptionWidth() const {
  return std::strlen(ArgStr)+6;
}

void Option::printOptionInfo(unsigned GlobalWidth) const {
  unsigned L = std::strlen(ArgStr);
  if (L == 0) return;  // Don't print the empty arg like this!
  cerr << "  -" << ArgStr << string(GlobalWidth-L-6, ' ') << " - "
       << HelpStr << "\n";
}


//===----------------------------------------------------------------------===//
// Boolean/flag command line option implementation
//

bool Flag::handleOccurance(const char *ArgName, const string &Arg) {
  if (Arg == "" || Arg == "true" || Arg == "TRUE" || Arg == "True" || 
      Arg == "1") {
    Value = true;
  } else if (Arg == "false" || Arg == "FALSE" || Arg == "False" || Arg == "0") {
    Value = false;
  } else {
    return error(": '" + Arg +
		 "' is invalid value for boolean argument! Try 0 or 1");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Integer valued command line option implementation
//
bool Int::handleOccurance(const char *ArgName, const string &Arg) {
  const char *ArgStart = Arg.c_str();
  char *End;
  Value = (int)strtol(ArgStart, &End, 0);
  if (*End != 0) 
    return error(": '" + Arg + "' value invalid for integer argument!");
  return false;  
}

//===----------------------------------------------------------------------===//
// String valued command line option implementation
//
bool String::handleOccurance(const char *ArgName, const string &Arg) {
  *this = Arg;
  return false;
}

//===----------------------------------------------------------------------===//
// StringList valued command line option implementation
//
bool StringList::handleOccurance(const char *ArgName, const string &Arg) {
  push_back(Arg);
  return false;
}

//===----------------------------------------------------------------------===//
// Enum valued command line option implementation
//
void EnumBase::processValues(va_list Vals) {
  while (const char *EnumName = va_arg(Vals, const char *)) {
    int EnumVal = va_arg(Vals, int);
    const char *EnumDesc = va_arg(Vals, const char *);
    ValueMap.push_back(std::make_pair(EnumName,      // Add value to value map
                                      std::make_pair(EnumVal, EnumDesc)));
  }
}

// registerArgs - notify the system about these new arguments
void EnumBase::registerArgs() {
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    AddArgument(ValueMap[i].first, this);
}

const char *EnumBase::getArgName(int ID) const {
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    if (ID == ValueMap[i].second.first) return ValueMap[i].first;
  return "";
}
const char *EnumBase::getArgDescription(int ID) const {
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    if (ID == ValueMap[i].second.first) return ValueMap[i].second.second;
  return "";
}



bool EnumValueBase::handleOccurance(const char *ArgName, const string &Arg) {
  unsigned i;
  for (i = 0; i < ValueMap.size(); ++i)
    if (ValueMap[i].first == Arg) break;

  if (i == ValueMap.size()) {
    string Alternatives;
    for (i = 0; i < ValueMap.size(); ++i) {
      if (i) Alternatives += ", ";
      Alternatives += ValueMap[i].first;
    }

    return error(": unrecognized alternative '" + Arg +
                 "'!  Alternatives are: " + Alternatives);
  }
  setValue(ValueMap[i].second.first);
  return false;
}

// Return the width of the option tag for printing...
unsigned EnumValueBase::getOptionWidth() const {
  unsigned BaseSize = Option::getOptionWidth();
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    BaseSize = std::max(BaseSize, (unsigned)std::strlen(ValueMap[i].first)+8);
  return BaseSize;
}

// printOptionInfo - Print out information about this option.  The 
// to-be-maintained width is specified.
//
void EnumValueBase::printOptionInfo(unsigned GlobalWidth) const {
  Option::printOptionInfo(GlobalWidth);
  for (unsigned i = 0; i < ValueMap.size(); ++i) {
    unsigned NumSpaces = GlobalWidth-strlen(ValueMap[i].first)-8;
    cerr << "    =" << ValueMap[i].first << string(NumSpaces, ' ') << " - "
	 << ValueMap[i].second.second;

    if (i == 0) cerr << " (default)";
    cerr << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Enum flags command line option implementation
//

bool EnumFlagsBase::handleOccurance(const char *ArgName, const string &Arg) {
  return EnumValueBase::handleOccurance("", ArgName);
}

unsigned EnumFlagsBase::getOptionWidth() const {
  unsigned BaseSize = 0;
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    BaseSize = std::max(BaseSize, (unsigned)std::strlen(ValueMap[i].first)+6);
  return BaseSize;
}

void EnumFlagsBase::printOptionInfo(unsigned GlobalWidth) const {
  for (unsigned i = 0; i < ValueMap.size(); ++i) {
    unsigned L = std::strlen(ValueMap[i].first);
    cerr << "  -" << ValueMap[i].first << string(GlobalWidth-L-6, ' ') << " - "
	 << ValueMap[i].second.second;
    if (i == 0) cerr << " (default)";
    cerr << "\n";
  }
}


//===----------------------------------------------------------------------===//
// Enum list command line option implementation
//

bool EnumListBase::handleOccurance(const char *ArgName, const string &Arg) {
  unsigned i;
  for (i = 0; i < ValueMap.size(); ++i)
    if (ValueMap[i].first == string(ArgName)) break;
  if (i == ValueMap.size())
    return error(": CommandLine INTERNAL ERROR", ArgName);
  Values.push_back(ValueMap[i].second.first);
  return false;
}

// Return the width of the option tag for printing...
unsigned EnumListBase::getOptionWidth() const {
  unsigned BaseSize = 0;
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    BaseSize = std::max(BaseSize, (unsigned)std::strlen(ValueMap[i].first)+6);
  return BaseSize;
}


// printOptionInfo - Print out information about this option.  The 
// to-be-maintained width is specified.
//
void EnumListBase::printOptionInfo(unsigned GlobalWidth) const {
  for (unsigned i = 0; i < ValueMap.size(); ++i) {
    unsigned L = std::strlen(ValueMap[i].first);
    cerr << "  -" << ValueMap[i].first << string(GlobalWidth-L-6, ' ') << " - "
	 << ValueMap[i].second.second << "\n";
  }
}


//===----------------------------------------------------------------------===//
// Help option... always automatically provided.
//
namespace {

// isHidden/isReallyHidden - Predicates to be used to filter down arg lists.
inline bool isHidden(pair<string, Option *> &OptPair) {
  return OptPair.second->getOptionHiddenFlag() >= Hidden;
}
inline bool isReallyHidden(pair<string, Option *> &OptPair) {
  return OptPair.second->getOptionHiddenFlag() == ReallyHidden;
}

class Help : public Option {
  unsigned MaxArgLen;
  const Option *EmptyArg;
  const bool ShowHidden;

  virtual bool handleOccurance(const char *ArgName, const string &Arg) {
    // Copy Options into a vector so we can sort them as we like...
    vector<pair<string, Option*> > Options;
    copy(getOpts().begin(), getOpts().end(), std::back_inserter(Options));

    // Eliminate Hidden or ReallyHidden arguments, depending on ShowHidden
    Options.erase(remove_if(Options.begin(), Options.end(), 
			  std::ptr_fun(ShowHidden ? isReallyHidden : isHidden)),
		  Options.end());

    // Eliminate duplicate entries in table (from enum flags options, f.e.)
    std::set<Option*> OptionSet;
    for (unsigned i = 0; i < Options.size(); )
      if (OptionSet.count(Options[i].second) == 0)
	OptionSet.insert(Options[i++].second); // Add to set
      else
	Options.erase(Options.begin()+i);      // Erase duplicate


    if (ProgramOverview)
      cerr << "OVERVIEW:" << ProgramOverview << "\n";
    // TODO: Sort options by some criteria

    cerr << "USAGE: " << ProgramName << " [options]\n\n";
    // TODO: print usage nicer

    // Compute the maximum argument length...
    MaxArgLen = 0;
    for_each(Options.begin(), Options.end(),
	     bind_obj(this, &Help::getMaxArgLen));

    cerr << "OPTIONS:\n";
    for_each(Options.begin(), Options.end(), 
	     bind_obj(this, &Help::printOption));

    return true;  // Displaying help is cause to terminate the program
  }

  void getMaxArgLen(pair<string, Option *> OptPair) {
    const Option *Opt = OptPair.second;
    if (Opt->ArgStr[0] == 0) EmptyArg = Opt; // Capture the empty arg if exists
    MaxArgLen = std::max(MaxArgLen, Opt->getOptionWidth());
  }

  void printOption(pair<string, Option *> OptPair) {
    const Option *Opt = OptPair.second;
    Opt->printOptionInfo(MaxArgLen);
  }

public:
  inline Help(const char *ArgVal, const char *HelpVal, bool showHidden)
    : Option(ArgVal, HelpVal, showHidden ? Hidden : 0), ShowHidden(showHidden) {
    EmptyArg = 0;
  }
};

Help HelpOp("help", "display available options"
	    " (--help-hidden for more)", false);
Help HelpHiddenOpt("help-hidden", "display all available options", true);

} // End anonymous namespace
