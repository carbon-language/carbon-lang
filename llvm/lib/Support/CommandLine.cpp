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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/STLExtras.h"
#include <vector>
#include <algorithm>
#include <map>
#include <set>
using namespace cl;

// Return the global command line option vector.  Making it a function scoped
// static ensures that it will be initialized before its first use correctly.
//
static map<string, Option*> &getOpts() {
  static map<string,Option*> CommandLineOptions;
  return CommandLineOptions;
}

static void AddArgument(const string &ArgName, Option *Opt) {
  if (getOpts().find(ArgName) != getOpts().end()) {
    cerr << "CommandLine Error: Argument '" << ArgName
	 << "' specified more than once!\n";
  } else {
    getOpts()[ArgName] = Opt;  // Add argument to the argument map!
  }
}

static const char *ProgramName = 0;
static const char *ProgramOverview = 0;

void cl::ParseCommandLineOptions(int &argc, char **argv,
				 const char *Overview = 0) {
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
      while (*ArgNameEnd && *ArgNameEnd != '=' &&
             *ArgNameEnd != '/') ++ArgNameEnd; // Scan till end

      Value = ArgNameEnd;
      if (*Value)           // If we have an equals sign...
	++Value;            // Advance to value...

      if (*ArgName != 0) {
	// Extract arg name part
        map<string, Option*>::iterator I = getOpts().find(string(ArgName, ArgNameEnd));
        Handler = I != getOpts().end() ? I->second : 0;
      }
    }

    if (Handler == 0) {
      cerr << "Unknown command line argument '" << argv[i] << "'.  Try: "
	   << argv[0] << " --help'\n";
      ErrorParsing = true;
      continue;
    }

    // Enforce value requirements
    switch (Handler->getValueExpectedFlag()) {
    case ValueRequired:
      if (Value == 0 || *Value == 0) {  // No value specified?
	if (i+1 < argc) {     // Steal the next argument, like for '-o filename'
	  Value = argv[++i];
	} else {
	  ErrorParsing = Handler->error(" requires a value!");
	  continue;
	}
      }
      break;
    case ValueDisallowed:
      if (*Value != 0) {
	ErrorParsing = Handler->error(" does not allow a value! '" + 
				      string(Value) + "' specified.");
	continue;
      }
      break;
    case ValueOptional: break;
    default: cerr << "Bad ValueMask flag! CommandLine usage error:" 
		  << Handler->getValueExpectedFlag() << endl; abort();
    }

    // Run the handler now!
    ErrorParsing |= Handler->addOccurance(ArgName, Value);
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
  cerr << "-" << ArgName << " option" << Message << endl;
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
  case ZeroOrMore: break;
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
       << HelpStr << endl;
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
  Values.push_back(Arg);
  return false;
}

//===----------------------------------------------------------------------===//
// Enum valued command line option implementation
//
void EnumBase::processValues(va_list Vals) {
  while (const char *EnumName = va_arg(Vals, const char *)) {
    int EnumVal = va_arg(Vals, int);
    const char *EnumDesc = va_arg(Vals, const char *);
    ValueMap.push_back(make_pair(EnumName,           // Add value to value map
				 make_pair(EnumVal, EnumDesc)));
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
  if (i == ValueMap.size())
    return error(": unrecognized alternative '"+Arg+"'!");
  Value = ValueMap[i].second.first;
  return false;
}

// Return the width of the option tag for printing...
unsigned EnumValueBase::getOptionWidth() const {
  unsigned BaseSize = Option::getOptionWidth();
  for (unsigned i = 0; i < ValueMap.size(); ++i)
    BaseSize = max(BaseSize, std::strlen(ValueMap[i].first)+8);
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
    cerr << endl;
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
    BaseSize = max(BaseSize, std::strlen(ValueMap[i].first)+6);
  return BaseSize;
}

void EnumFlagsBase::printOptionInfo(unsigned GlobalWidth) const {
  for (unsigned i = 0; i < ValueMap.size(); ++i) {
    unsigned L = std::strlen(ValueMap[i].first);
    cerr << "  -" << ValueMap[i].first << string(GlobalWidth-L-6, ' ') << " - "
	 << ValueMap[i].second.second;
    if (i == 0) cerr << " (default)";
    cerr << endl;
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
    BaseSize = max(BaseSize, std::strlen(ValueMap[i].first)+6);
  return BaseSize;
}


// printOptionInfo - Print out information about this option.  The 
// to-be-maintained width is specified.
//
void EnumListBase::printOptionInfo(unsigned GlobalWidth) const {
  for (unsigned i = 0; i < ValueMap.size(); ++i) {
    unsigned L = std::strlen(ValueMap[i].first);
    cerr << "  -" << ValueMap[i].first << string(GlobalWidth-L-6, ' ') << " - "
	 << ValueMap[i].second.second << endl;
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
    copy(getOpts().begin(), getOpts().end(), back_inserter(Options));

    // Eliminate Hidden or ReallyHidden arguments, depending on ShowHidden
    Options.erase(remove_if(Options.begin(), Options.end(), 
			    ptr_fun(ShowHidden ? isReallyHidden : isHidden)),
		  Options.end());

    // Eliminate duplicate entries in table (from enum flags options, f.e.)
    set<Option*> OptionSet;
    for (unsigned i = 0; i < Options.size(); )
      if (OptionSet.count(Options[i].second) == 0)
	OptionSet.insert(Options[i++].second); // Add to set
      else
	Options.erase(Options.begin()+i);      // Erase duplicate


    if (ProgramOverview)
      cerr << "OVERVIEW:" << ProgramOverview << endl;
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
    MaxArgLen = max(MaxArgLen, Opt->getOptionWidth());
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
