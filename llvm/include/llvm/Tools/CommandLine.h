//===-- llvm/Tools/CommandLine.h - Command line parser for tools -*- C++ -*--=//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_COMMANDLINE_H
#define LLVM_TOOLS_COMMANDLINE_H

#include <string>
#include <vector>
#include <utility>
#include <stdarg.h>

namespace cl {   // Short namespace to make usage concise

//===----------------------------------------------------------------------===//
// ParseCommandLineOptions - Minimalistic command line option processing entry
//
void cl::ParseCommandLineOptions(int &argc, char **argv,
				 const char *Overview = 0);


//===----------------------------------------------------------------------===//
// Global flags permitted to be passed to command line arguments

enum FlagsOptions {
  NoFlags         = 0x00,      // Marker to make explicit that we have no flags

  // Flags for the number of occurances allowed...
  Optional        = 0x00,      // Zero or One occurance
  ZeroOrMore      = 0x01,      // Zero or more occurances allowed
  Required        = 0x02,      // One occurance required
  OneOrMore       = 0x03,      // One or more occurances required
  OccurancesMask  = 0x07,

  // Number of arguments to a value expected...
  //Optional      = 0x00,      // The value can oppear... or not
  ValueRequired   = 0x08,      // The value is required to appear!
  ValueDisallowed = 0x10,      // A value may not be specified (for flags)
  ValueMask       = 0x18,

  // Control whether -help shows the command line option...
  Hidden          = 0x20,      // -help doesn't -help-hidden does
  ReallyHidden    = 0x60,      // Neither -help nor -help-hidden show this arg
  HiddenMask      = 0x60,
};


//===----------------------------------------------------------------------===//
// Option Base class
//
class Option {
  friend void cl::ParseCommandLineOptions(int &, char **, const char *Overview);

  // handleOccurances - Overriden by subclasses to handle the value passed into
  // an argument.  Should return true if there was an error processing the
  // argument and the program should exit.
  //
  virtual bool handleOccurance(const char *ArgName, const string &Arg) = 0;

  int NumOccurances;          // The number of times specified
public:
  const char * const ArgStr;  // The argument string itself (ex: "help", "o")
  const char * const HelpStr; // The descriptive text message for --help
  const int Flags;            // Flags for the argument

protected:
  Option(const char *ArgStr, const char *Message, int Flags);
  Option(int flags) : ArgStr(""), HelpStr(""), Flags(flags) {}

  // Prints option name followed by message.  Always returns true.
  bool error(string Message, const char *ArgName = 0);

  // addOccurance - Wrapper around handleOccurance that enforces Flags
  //
  bool addOccurance(const char *ArgName, const string &Value);

public:
  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;

public:
  inline int getNumOccurances() const { return NumOccurances; }
  virtual ~Option() {}
};


//===----------------------------------------------------------------------===//
// Boolean/flag command line option
//
class Flag : public Option {
  bool Value;
  virtual bool handleOccurance(const char *ArgName, const string &Arg);
public:
  inline Flag(const char *ArgStr, const char *Message, int Flags = 0, 
	      bool DefaultVal = 0) : Option(ArgStr, Message, Flags), 
				     Value(DefaultVal) {}
  operator bool() const { return Value; }
  inline bool getValue() const { return Value; }
  inline void setValue(bool Val) { Value = Val; }
};



//===----------------------------------------------------------------------===//
// Integer valued command line option
//
class Int : public Option {
  int Value;
  virtual bool handleOccurance(const char *ArgName, const string &Arg);
public:
  inline Int(const char *ArgStr, const char *Help, int Flags = 0,
	     int DefaultVal = 0) : Option(ArgStr, Help, Flags | ValueRequired),
				   Value(DefaultVal) {}
  inline operator int() const { return Value; }
  inline int getValue() const { return Value; }
  inline void setValue(int Val) { Value = Val; }
};


//===----------------------------------------------------------------------===//
// String valued command line option
//
class String : public Option {
  string Value;
  virtual bool handleOccurance(const char *ArgName, const string &Arg);
public:
  inline String(const char *ArgStr, const char *Help, int Flags = 0, 
		const char *DefaultVal = "") 
    : Option(ArgStr, Help, Flags | ValueRequired), Value(DefaultVal) {}

  inline const string &getValue() const { return Value; }
  inline void setValue(const string &Val) { Value = Val; }
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
  vector<pair<const char *, pair<int, const char *> > > ValueMap;

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
  int Value;
  inline EnumValueBase(const char *ArgStr, const char *Help, int Flags)
    : EnumBase(ArgStr, Help, Flags) {}
  inline EnumValueBase(int Flags) : EnumBase(Flags) {}

  // handleOccurance - Set Value to the enum value specified by Arg
  virtual bool handleOccurance(const char *ArgName, const string &Arg);

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;
};

template <class E>  // The enum we are representing
class Enum : public EnumValueBase {
public:
  inline Enum(const char *ArgStr, int Flags, const char *Help, ...)
    : EnumValueBase(ArgStr, Help, Flags | ValueRequired) {
    va_list Values;
    va_start(Values, Help);
    processValues(Values);
    va_end(Values);
    Value = ValueMap.front().second.first; // Grab default value
  }

  inline E getValue() const { return (E)Value; }
  inline void setValue(E Val) { Value = (E)Val; }
};


//===----------------------------------------------------------------------===//
// Enum flags command line option
//
class EnumFlagsBase : public EnumValueBase {
protected:
  virtual bool handleOccurance(const char *ArgName, const string &Arg);
  inline EnumFlagsBase(int Flags) : EnumValueBase(Flags | ValueDisallowed) {}

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const;
};

template <class E>  // The enum we are representing
class EnumFlags : public EnumFlagsBase {
public:
  inline EnumFlags(int Flags, ...) : EnumFlagsBase(Flags) {
    va_list Values;
    va_start(Values, Flags);
    processValues(Values);
    va_end(Values);
    registerArgs();
    Value = ValueMap.front().second.first; // Grab default value
  }
  inline E getValue() const { return (E)Value; }
  inline void setValue(E Val) { Value = (E)Val; }
};


//===----------------------------------------------------------------------===//
// Enum list command line option
//
class EnumListBase : public EnumBase {
protected:
  vector<int> Values;  // The options specified so far.

  inline EnumListBase(int Flags) 
    : EnumBase(Flags | ValueDisallowed | ZeroOrMore) {}
  virtual bool handleOccurance(const char *ArgName, const string &Arg);

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
  inline E getValue(unsigned i) const { return (E)Values[i]; }
  inline E operator[](unsigned i) const { return (E)Values[i]; }
};

} // End namespace cl

#endif
