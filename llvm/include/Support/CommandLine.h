//===- Support/CommandLine.h - Flexible Command line parser ------*- C++ -*--=//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
// Note that rather than trying to figure out what this code does, you could try
// reading the library documentation located in docs/CommandLine.html
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMMANDLINE_H
#define LLVM_SUPPORT_COMMANDLINE_H

#include <string>
#include <vector>
#include <utility>
#include <stdarg.h>

namespace cl {   // Short namespace to make usage concise

//===----------------------------------------------------------------------===//
// ParseCommandLineOptions - Minimalistic command line option processing entry
//
void cl::ParseCommandLineOptions(int &argc, char **argv,
				 const char *Overview = 0,
				 int Flags = 0);

// ParserOptions - This set of option is use to control global behavior of the
// command line processor.
//
enum ParserOptions {
  // DisableSingleLetterArgGrouping - With this option enabled, multiple letter
  // options are allowed to bunch together with only a single hyphen for the
  // whole group.  This allows emulation of the behavior that ls uses for
  // example:  ls -la === ls -l -a    Providing this option, disables this.
  //
  DisableSingleLetterArgGrouping = 0x0001,

  // EnableSingleLetterArgValue - This option allows arguments that are
  // otherwise unrecognized to match single letter flags that take a value. 
  // This is useful for cases like a linker, where options are typically of the
  // form '-lfoo' or '-L../../include' where -l or -L are the actual flags.
  //
  EnableSingleLetterArgValue     = 0x0002,
};


//===----------------------------------------------------------------------===//
// Global flags permitted to be passed to command line arguments

enum FlagsOptions {
  NoFlags         = 0x00,      // Marker to make explicit that we have no flags
  Default         = 0x00,      // Equally, marker to use the default flags

  GlobalsMask     = 0x80, 
};

enum NumOccurances {           // Flags for the number of occurances allowed...
  Optional        = 0x01,      // Zero or One occurance
  ZeroOrMore      = 0x02,      // Zero or more occurances allowed
  Required        = 0x03,      // One occurance required
  OneOrMore       = 0x04,      // One or more occurances required

  // ConsumeAfter - Marker for a null ("") flag that can be used to indicate
  // that anything that matches the null marker starts a sequence of options
  // that all get sent to the null marker.  Thus, for example, all arguments
  // to LLI are processed until a filename is found.  Once a filename is found,
  // all of the succeeding arguments are passed, unprocessed, to the null flag.
  //
  ConsumeAfter    = 0x05,

  OccurancesMask  = 0x07,
};

enum ValueExpected {           // Is a value required for the option?
  ValueOptional   = 0x08,      // The value can oppear... or not
  ValueRequired   = 0x10,      // The value is required to appear!
  ValueDisallowed = 0x18,      // A value may not be specified (for flags)
  ValueMask       = 0x18,
};

enum OptionHidden {            // Control whether -help shows this option
  NotHidden       = 0x20,      // Option included in --help & --help-hidden
  Hidden          = 0x40,      // -help doesn't, but --help-hidden does
  ReallyHidden    = 0x60,      // Neither --help nor --help-hidden show this arg
  HiddenMask      = 0x60,
};


//===----------------------------------------------------------------------===//
// Option Base class
//
class Alias;
class Option {
  friend void cl::ParseCommandLineOptions(int &, char **, const char *, int);
  friend class Alias;

  // handleOccurances - Overriden by subclasses to handle the value passed into
  // an argument.  Should return true if there was an error processing the
  // argument and the program should exit.
  //
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg) = 0;

  virtual enum NumOccurances getNumOccurancesFlagDefault() const { 
    return Optional;
  }
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional; 
  }
  virtual enum OptionHidden getOptionHiddenFlagDefault() const {
    return NotHidden;
  }

  int NumOccurances;          // The number of times specified
  const int Flags;            // Flags for the argument
public:
  const char * const ArgStr;  // The argument string itself (ex: "help", "o")
  const char * const HelpStr; // The descriptive text message for --help

  inline enum NumOccurances getNumOccurancesFlag() const {
    int NO = Flags & OccurancesMask;
    return NO ? (enum NumOccurances)NO : getNumOccurancesFlagDefault();
  }
  inline enum ValueExpected getValueExpectedFlag() const {
    int VE = Flags & ValueMask;
    return VE ? (enum ValueExpected)VE : getValueExpectedFlagDefault();
  }
  inline enum OptionHidden getOptionHiddenFlag() const {
    int OH = Flags & HiddenMask;
    return OH ? (enum OptionHidden)OH : getOptionHiddenFlagDefault();
  }

protected:
  Option(const char *ArgStr, const char *Message, int Flags);
  Option(int flags) : NumOccurances(0), Flags(flags), ArgStr(""), HelpStr("") {}

public:
  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;

  // addOccurance - Wrapper around handleOccurance that enforces Flags
  //
  bool addOccurance(const char *ArgName, const std::string &Value);

  // Prints option name followed by message.  Always returns true.
  bool error(std::string Message, const char *ArgName = 0);

public:
  inline int getNumOccurances() const { return NumOccurances; }
  virtual ~Option() {}
};


//===----------------------------------------------------------------------===//
// Aliased command line option (alias this name to a preexisting name)
//
class Alias : public Option {
  Option &AliasFor;
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg) {
    return AliasFor.handleOccurance(AliasFor.ArgStr, Arg);
  }
  virtual enum OptionHidden getOptionHiddenFlagDefault() const {return Hidden;}
public:
  inline Alias(const char *ArgStr, const char *Message, int Flags,
	       Option &aliasFor) : Option(ArgStr, Message, Flags), 
				   AliasFor(aliasFor) {}
};

//===----------------------------------------------------------------------===//
// Boolean/flag command line option
//
class Flag : public Option {
  bool &Value;
  bool DValue;
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);
public:
  inline Flag(const char *ArgStr, const char *Message, int Flags = 0, 
	      bool DefaultVal = false)
    : Option(ArgStr, Message, Flags), Value(DValue) {
    Value = DefaultVal;
  }

  inline Flag(bool &UpdateVal, const char *ArgStr, const char *Message,
              int Flags = 0, bool DefaultVal = false)
    : Option(ArgStr, Message, Flags), Value(UpdateVal) {
    Value = DefaultVal;
  }

  operator const bool() const { return Value; }
  inline bool operator=(bool Val) { Value = Val; return Val; }
};



//===----------------------------------------------------------------------===//
// Integer valued command line option
//
class Int : public Option {
  int Value;
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired; 
  }
public:
  inline Int(const char *ArgStr, const char *Help, int Flags = 0,
	     int DefaultVal = 0) : Option(ArgStr, Help, Flags),
				   Value(DefaultVal) {}
  inline operator int() const { return Value; }
  inline int operator=(int Val) { Value = Val; return Val; }
};


//===----------------------------------------------------------------------===//
// String valued command line option
//
class String : public Option, public std::string {
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired; 
  }
public:
  inline String(const char *ArgStr, const char *Help, int Flags = 0, 
		const char *DefaultVal = "") 
    : Option(ArgStr, Help, Flags), std::string(DefaultVal) {}

  inline const std::string &operator=(const std::string &Val) { 
    return std::string::operator=(Val);
  }
};


//===----------------------------------------------------------------------===//
// String list command line option
//
class StringList : public Option, public std::vector<std::string> {

  virtual enum NumOccurances getNumOccurancesFlagDefault() const { 
    return ZeroOrMore;
  }
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired;
  }
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);

public:
  inline StringList(const char *ArgStr, const char *Help, int Flags = 0)
    : Option(ArgStr, Help, Flags) {}
};


//===----------------------------------------------------------------------===//
// Enum valued command line option
//
#define clEnumVal(ENUMVAL, DESC) #ENUMVAL, ENUMVAL, DESC
#define clEnumValN(ENUMVAL, FLAGNAME, DESC) FLAGNAME, ENUMVAL, DESC

// EnumBase - Base class for all enum/varargs related argument types...
class EnumBase : public Option {
protected:
  // Use a vector instead of a map, because the lists should be short,
  // the overhead is less, and most importantly, it keeps them in the order
  // inserted so we can print our option out nicely.
  std::vector<std::pair<const char *, std::pair<int, const char *> > > ValueMap;

  inline EnumBase(const char *ArgStr, const char *Help, int Flags)
    : Option(ArgStr, Help, Flags) {}
  inline EnumBase(int Flags) : Option(Flags) {}

  // processValues - Incorporate the specifed varargs arglist into the 
  // ValueMap.
  //
  void processValues(va_list Vals);

  // registerArgs - notify the system about these new arguments
  void registerArgs();

public:
  // Turn an enum into the arg name that activates it
  const char *getArgName(int ID) const;
  const char *getArgDescription(int ID) const;
};

class EnumValueBase : public EnumBase {
protected:
  inline EnumValueBase(const char *ArgStr, const char *Help, int Flags)
    : EnumBase(ArgStr, Help, Flags) {}
  inline EnumValueBase(int Flags) : EnumBase(Flags) {}

  // handleOccurance - Set Value to the enum value specified by Arg
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;

  // setValue - Subclasses override this when they need to receive a new value
  virtual void setValue(int Val) = 0;
};

template <class E>  // The enum we are representing
class Enum : public EnumValueBase {
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired;
  }
  E DVal;
  E &Value;

  // setValue - Subclasses override this when they need to receive a new value
  virtual void setValue(int Val) { Value = (E)Val; }
public:
  inline Enum(const char *ArgStr, int Flags, const char *Help, ...)
    : EnumValueBase(ArgStr, Help, Flags), Value(DVal) {
    va_list Values;
    va_start(Values, Help);
    processValues(Values);
    va_end(Values);
    Value = (E)ValueMap.front().second.first; // Grab default value
  }

  inline Enum(E &EUpdate, const char *ArgStr, int Flags, const char *Help, ...)
    : EnumValueBase(ArgStr, Help, Flags), Value(EUpdate) {
    va_list Values;
    va_start(Values, Help);
    processValues(Values);
    va_end(Values);
    Value = (E)ValueMap.front().second.first; // Grab default value
  }

  inline operator E() const { return Value; }
  inline E operator=(E Val) { Value = Val; return Val; }
};


//===----------------------------------------------------------------------===//
// Enum flags command line option
//
class EnumFlagsBase : public EnumValueBase {
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueDisallowed;
  }
protected:
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);
  inline EnumFlagsBase(int Flags) : EnumValueBase(Flags) {}

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;
};

template <class E>  // The enum we are representing
class EnumFlags : public EnumFlagsBase {
  E DVal;
  E &Value;

  // setValue - Subclasses override this when they need to receive a new value
  virtual void setValue(int Val) { Value = (E)Val; }
public:
  inline EnumFlags(int Flags, ...) : EnumFlagsBase(Flags), Value(DVal) {
    va_list Values;
    va_start(Values, Flags);
    processValues(Values);
    va_end(Values);
    registerArgs();
    Value = (E)ValueMap.front().second.first; // Grab default value
  }
  inline EnumFlags(E &RV, int Flags, ...) : EnumFlagsBase(Flags), Value(RV) {
    va_list Values;
    va_start(Values, Flags);
    processValues(Values);
    va_end(Values);
    registerArgs();
    Value = (E)ValueMap.front().second.first; // Grab default value
  }

  inline operator E() const { return (E)Value; }
  inline E operator=(E Val) { Value = Val; return Val; }
};


//===----------------------------------------------------------------------===//
// Enum list command line option
//
class EnumListBase : public EnumBase {
  virtual enum NumOccurances getNumOccurancesFlagDefault() const { 
    return ZeroOrMore;
  }
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueDisallowed;
  }
protected:
  std::vector<int> Values;  // The options specified so far.

  inline EnumListBase(int Flags) 
    : EnumBase(Flags) {}
  virtual bool handleOccurance(const char *ArgName, const std::string &Arg);

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;
public:
  inline unsigned size() { return Values.size(); }
};

template <class E>  // The enum we are representing
class EnumList : public EnumListBase {
public:
  inline EnumList(int Flags, ...) : EnumListBase(Flags) {
    va_list Values;
    va_start(Values, Flags);
    processValues(Values);
    va_end(Values);
    registerArgs();
  }
  inline E  operator[](unsigned i) const { return (E)Values[i]; }
  inline E &operator[](unsigned i)       { return (E&)Values[i]; }
};

} // End namespace cl

#endif
