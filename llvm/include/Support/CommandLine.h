//===- Support/CommandLine.h - Flexible Command line parser -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This class implements a command line argument processor that is useful when
// creating a tool.  It provides a simple, minimalistic interface that is easily
// extensible and supports nonlocal (library) command line options.
//
// Note that rather than trying to figure out what this code does, you should
// read the library documentation located in docs/CommandLine.html or looks at
// the many example usages in tools/*/*.cpp
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMMANDLINE_H
#define SUPPORT_COMMANDLINE_H

#include "Support/type_traits.h"
#include <string>
#include <vector>
#include <utility>
#include <cstdarg>
#include <cassert>

namespace llvm {

/// cl Namespace - This namespace contains all of the command line option
/// processing machinery.  It is intentionally a short name to make qualified
/// usage concise.
namespace cl {

//===----------------------------------------------------------------------===//
// ParseCommandLineOptions - Command line option processing entry point.
//
void ParseCommandLineOptions(int &argc, char **argv,
                             const char *Overview = 0);

//===----------------------------------------------------------------------===//
// ParseEnvironmentOptions - Environment variable option processing alternate
//                           entry point.
//
void ParseEnvironmentOptions(const char *progName, const char *envvar,
                             const char *Overview = 0);

//===----------------------------------------------------------------------===//
// Flags permitted to be passed to command line arguments
//

enum NumOccurrences {           // Flags for the number of occurrences allowed
  Optional        = 0x01,      // Zero or One occurrence
  ZeroOrMore      = 0x02,      // Zero or more occurrences allowed
  Required        = 0x03,      // One occurrence required
  OneOrMore       = 0x04,      // One or more occurrences required

  // ConsumeAfter - Indicates that this option is fed anything that follows the
  // last positional argument required by the application (it is an error if
  // there are zero positional arguments, and a ConsumeAfter option is used).
  // Thus, for example, all arguments to LLI are processed until a filename is
  // found.  Once a filename is found, all of the succeeding arguments are
  // passed, unprocessed, to the ConsumeAfter option.
  //
  ConsumeAfter    = 0x05,

  OccurrencesMask  = 0x07,
};

enum ValueExpected {           // Is a value required for the option?
  ValueOptional   = 0x08,      // The value can appear... or not
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

// Formatting flags - This controls special features that the option might have
// that cause it to be parsed differently...
//
// Prefix - This option allows arguments that are otherwise unrecognized to be
// matched by options that are a prefix of the actual value.  This is useful for
// cases like a linker, where options are typically of the form '-lfoo' or
// '-L../../include' where -l or -L are the actual flags.  When prefix is
// enabled, and used, the value for the flag comes from the suffix of the
// argument.
//
// Grouping - With this option enabled, multiple letter options are allowed to
// bunch together with only a single hyphen for the whole group.  This allows
// emulation of the behavior that ls uses for example: ls -la === ls -l -a
//

enum FormattingFlags {
  NormalFormatting = 0x000,     // Nothing special
  Positional       = 0x080,     // Is a positional argument, no '-' required
  Prefix           = 0x100,     // Can this option directly prefix its value?
  Grouping         = 0x180,     // Can this option group with other options?
  FormattingMask   = 0x180,
};

enum MiscFlags {                // Miscellaneous flags to adjust argument
  CommaSeparated   = 0x200,     // Should this cl::list split between commas?
  MiscMask         = 0x200,
};



//===----------------------------------------------------------------------===//
// Option Base class
//
class alias;
class Option {
  friend void cl::ParseCommandLineOptions(int &, char **, const char *, int);
  friend class alias;

  // handleOccurrences - Overriden by subclasses to handle the value passed into
  // an argument.  Should return true if there was an error processing the
  // argument and the program should exit.
  //
  virtual bool handleOccurrence(const char *ArgName, const std::string &Arg) = 0;

  virtual enum NumOccurrences getNumOccurrencesFlagDefault() const { 
    return Optional;
  }
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional; 
  }
  virtual enum OptionHidden getOptionHiddenFlagDefault() const {
    return NotHidden;
  }
  virtual enum FormattingFlags getFormattingFlagDefault() const {
    return NormalFormatting;
  }

  int NumOccurrences;    // The number of times specified
  int Flags;            // Flags for the argument
public:
  const char *ArgStr;   // The argument string itself (ex: "help", "o")
  const char *HelpStr;  // The descriptive text message for --help
  const char *ValueStr; // String describing what the value of this option is

  inline enum NumOccurrences getNumOccurrencesFlag() const {
    int NO = Flags & OccurrencesMask;
    return NO ? static_cast<enum NumOccurrences>(NO)
              : getNumOccurrencesFlagDefault();
  }
  inline enum ValueExpected getValueExpectedFlag() const {
    int VE = Flags & ValueMask;
    return VE ? static_cast<enum ValueExpected>(VE)
              : getValueExpectedFlagDefault();
  }
  inline enum OptionHidden getOptionHiddenFlag() const {
    int OH = Flags & HiddenMask;
    return OH ? static_cast<enum OptionHidden>(OH)
              : getOptionHiddenFlagDefault();
  }
  inline enum FormattingFlags getFormattingFlag() const {
    int OH = Flags & FormattingMask;
    return OH ? static_cast<enum FormattingFlags>(OH)
              : getFormattingFlagDefault();
  }
  inline unsigned getMiscFlags() const {
    return Flags & MiscMask;
  }

  // hasArgStr - Return true if the argstr != ""
  bool hasArgStr() const { return ArgStr[0] != 0; }

  //-------------------------------------------------------------------------===
  // Accessor functions set by OptionModifiers
  //
  void setArgStr(const char *S) { ArgStr = S; }
  void setDescription(const char *S) { HelpStr = S; }
  void setValueStr(const char *S) { ValueStr = S; }

  void setFlag(unsigned Flag, unsigned FlagMask) {
    if (Flags & FlagMask) {
      error(": Specified two settings for the same option!");
      exit(1);
    }

    Flags |= Flag;
  }

  void setNumOccurrencesFlag(enum NumOccurrences Val) {
    setFlag(Val, OccurrencesMask);
  }
  void setValueExpectedFlag(enum ValueExpected Val) { setFlag(Val, ValueMask); }
  void setHiddenFlag(enum OptionHidden Val) { setFlag(Val, HiddenMask); }
  void setFormattingFlag(enum FormattingFlags V) { setFlag(V, FormattingMask); }
  void setMiscFlag(enum MiscFlags M) { setFlag(M, M); }
protected:
  Option() : NumOccurrences(0), Flags(0),
             ArgStr(""), HelpStr(""), ValueStr("") {}

public:
  // addArgument - Tell the system that this Option subclass will handle all
  // occurrences of -ArgStr on the command line.
  //
  void addArgument(const char *ArgStr);
  void removeArgument(const char *ArgStr);

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth() const = 0;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(unsigned GlobalWidth) const = 0;

  // addOccurrence - Wrapper around handleOccurrence that enforces Flags
  //
  bool addOccurrence(const char *ArgName, const std::string &Value);

  // Prints option name followed by message.  Always returns true.
  bool error(std::string Message, const char *ArgName = 0);

public:
  inline int getNumOccurrences() const { return NumOccurrences; }
  virtual ~Option() {}
};


//===----------------------------------------------------------------------===//
// Command line option modifiers that can be used to modify the behavior of
// command line option parsers...
//

// desc - Modifier to set the description shown in the --help output...
struct desc {
  const char *Desc;
  desc(const char *Str) : Desc(Str) {}
  void apply(Option &O) const { O.setDescription(Desc); }
};

// value_desc - Modifier to set the value description shown in the --help
// output...
struct value_desc {
  const char *Desc;
  value_desc(const char *Str) : Desc(Str) {}
  void apply(Option &O) const { O.setValueStr(Desc); }
};


// init - Specify a default (initial) value for the command line argument, if
// the default constructor for the argument type does not give you what you
// want.  This is only valid on "opt" arguments, not on "list" arguments.
//
template<class Ty>
struct initializer {
  const Ty &Init;
  initializer(const Ty &Val) : Init(Val) {}

  template<class Opt>
  void apply(Opt &O) const { O.setInitialValue(Init); }
};

template<class Ty>
initializer<Ty> init(const Ty &Val) {
  return initializer<Ty>(Val);
}


// location - Allow the user to specify which external variable they want to
// store the results of the command line argument processing into, if they don't
// want to store it in the option itself.
//
template<class Ty>
struct LocationClass {
  Ty &Loc;
  LocationClass(Ty &L) : Loc(L) {}

  template<class Opt>
  void apply(Opt &O) const { O.setLocation(O, Loc); }
};

template<class Ty>
LocationClass<Ty> location(Ty &L) { return LocationClass<Ty>(L); }


//===----------------------------------------------------------------------===//
// Enum valued command line option
//
#define clEnumVal(ENUMVAL, DESC) #ENUMVAL, (int)ENUMVAL, DESC
#define clEnumValN(ENUMVAL, FLAGNAME, DESC) FLAGNAME, (int)ENUMVAL, DESC

// values - For custom data types, allow specifying a group of values together
// as the values that go into the mapping that the option handler uses.  Note
// that the values list must always have a 0 at the end of the list to indicate
// that the list has ended.
//
template<class DataType>
class ValuesClass {
  // Use a vector instead of a map, because the lists should be short,
  // the overhead is less, and most importantly, it keeps them in the order
  // inserted so we can print our option out nicely.
  std::vector<std::pair<const char *, std::pair<int, const char *> > > Values;
  void processValues(va_list Vals);
public:
  ValuesClass(const char *EnumName, DataType Val, const char *Desc, 
              va_list ValueArgs) {
    // Insert the first value, which is required.
    Values.push_back(std::make_pair(EnumName, std::make_pair(Val, Desc)));

    // Process the varargs portion of the values...
    while (const char *EnumName = va_arg(ValueArgs, const char *)) {
      DataType EnumVal = static_cast<DataType>(va_arg(ValueArgs, int));
      const char *EnumDesc = va_arg(ValueArgs, const char *);
      Values.push_back(std::make_pair(EnumName,      // Add value to value map
                                      std::make_pair(EnumVal, EnumDesc)));
    }
  }

  template<class Opt>
  void apply(Opt &O) const {
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      O.getParser().addLiteralOption(Values[i].first, Values[i].second.first,
                                     Values[i].second.second);
  }
};

template<class DataType>
ValuesClass<DataType> values(const char *Arg, DataType Val, const char *Desc,
                             ...) {
    va_list ValueArgs;
    va_start(ValueArgs, Desc);
    ValuesClass<DataType> Vals(Arg, Val, Desc, ValueArgs);
    va_end(ValueArgs);
    return Vals;
}


//===----------------------------------------------------------------------===//
// parser class - Parameterizable parser for different data types.  By default,
// known data types (string, int, bool) have specialized parsers, that do what
// you would expect.  The default parser, used for data types that are not
// built-in, uses a mapping table to map specific options to values, which is
// used, among other things, to handle enum types.

//--------------------------------------------------
// generic_parser_base - This class holds all the non-generic code that we do
// not need replicated for every instance of the generic parser.  This also
// allows us to put stuff into CommandLine.cpp
//
struct generic_parser_base {
  virtual ~generic_parser_base() {}  // Base class should have virtual-dtor

  // getNumOptions - Virtual function implemented by generic subclass to
  // indicate how many entries are in Values.
  //
  virtual unsigned getNumOptions() const = 0;

  // getOption - Return option name N.
  virtual const char *getOption(unsigned N) const = 0;
  
  // getDescription - Return description N
  virtual const char *getDescription(unsigned N) const = 0;

  // Return the width of the option tag for printing...
  virtual unsigned getOptionWidth(const Option &O) const;

  // printOptionInfo - Print out information about this option.  The 
  // to-be-maintained width is specified.
  //
  virtual void printOptionInfo(const Option &O, unsigned GlobalWidth) const;

  void initialize(Option &O) {
    // All of the modifiers for the option have been processed by now, so the
    // argstr field should be stable, copy it down now.
    //
    hasArgStr = O.hasArgStr();

    // If there has been no argstr specified, that means that we need to add an
    // argument for every possible option.  This ensures that our options are
    // vectored to us.
    //
    if (!hasArgStr)
      for (unsigned i = 0, e = getNumOptions(); i != e; ++i)
        O.addArgument(getOption(i));
  }

  enum ValueExpected getValueExpectedFlagDefault() const {
    // If there is an ArgStr specified, then we are of the form:
    //
    //    -opt=O2   or   -opt O2  or  -optO2
    //
    // In which case, the value is required.  Otherwise if an arg str has not
    // been specified, we are of the form:
    //
    //    -O2 or O2 or -la (where -l and -a are separate options)
    //
    // If this is the case, we cannot allow a value.
    //
    if (hasArgStr)
      return ValueRequired;
    else
      return ValueDisallowed;
  }

  // findOption - Return the option number corresponding to the specified
  // argument string.  If the option is not found, getNumOptions() is returned.
  //
  unsigned findOption(const char *Name);

protected:
  bool hasArgStr;
};

// Default parser implementation - This implementation depends on having a
// mapping of recognized options to values of some sort.  In addition to this,
// each entry in the mapping also tracks a help message that is printed with the
// command line option for --help.  Because this is a simple mapping parser, the
// data type can be any unsupported type.
//
template <class DataType>
class parser : public generic_parser_base {
protected:
  std::vector<std::pair<const char *,
                        std::pair<DataType, const char *> > > Values;
public:
  typedef DataType parser_data_type;

  // Implement virtual functions needed by generic_parser_base
  unsigned getNumOptions() const { return Values.size(); }
  const char *getOption(unsigned N) const { return Values[N].first; }
  const char *getDescription(unsigned N) const {
    return Values[N].second.second;
  }

  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg,
             DataType &V) {
    std::string ArgVal;
    if (hasArgStr)
      ArgVal = Arg;
    else
      ArgVal = ArgName;

    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      if (ArgVal == Values[i].first) {
        V = Values[i].second.first;
        return false;
      }

    return O.error(": Cannot find option named '" + ArgVal + "'!");
  }

  // addLiteralOption - Add an entry to the mapping table...
  template <class DT>
  void addLiteralOption(const char *Name, const DT &V, const char *HelpStr) {
    assert(findOption(Name) == Values.size() && "Option already exists!");
    Values.push_back(std::make_pair(Name,
                             std::make_pair(static_cast<DataType>(V),HelpStr)));
  }

  // removeLiteralOption - Remove the specified option.
  //
  void removeLiteralOption(const char *Name) {
    unsigned N = findOption(Name);
    assert(N != Values.size() && "Option not found!");
    Values.erase(Values.begin()+N);
  }
};

//--------------------------------------------------
// basic_parser - Super class of parsers to provide boilerplate code
//
struct basic_parser_impl {  // non-template implementation of basic_parser<t>
  virtual ~basic_parser_impl() {}

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueRequired;
  }
  
  void initialize(Option &O) {}
  
  // Return the width of the option tag for printing...
  unsigned getOptionWidth(const Option &O) const;
  
  // printOptionInfo - Print out information about this option.  The
  // to-be-maintained width is specified.
  //
  void printOptionInfo(const Option &O, unsigned GlobalWidth) const;


  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "value"; }
};

// basic_parser - The real basic parser is just a template wrapper that provides
// a typedef for the provided data type.
//
template<class DataType>
struct basic_parser : public basic_parser_impl {
  typedef DataType parser_data_type;
};


//--------------------------------------------------
// parser<bool>
//
template<>
struct parser<bool> : public basic_parser<bool> {

  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg, bool &Val);

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional; 
  }

  // getValueName - Do not print =<value> at all
  virtual const char *getValueName() const { return 0; }
};


//--------------------------------------------------
// parser<int>
//
template<>
struct parser<int> : public basic_parser<int> {
  
  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg, int &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "int"; }
};


//--------------------------------------------------
// parser<unsigned>
//
template<>
struct parser<unsigned> : public basic_parser<unsigned> {
  
  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg,
             unsigned &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "uint"; }
};


//--------------------------------------------------
// parser<double>
//
template<>
struct parser<double> : public basic_parser<double> {
  // parse - Return true on error.
  bool parse(Option &O, const char *AN, const std::string &Arg, double &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "number"; }
};


//--------------------------------------------------
// parser<float>
//
template<>
struct parser<float> : public basic_parser<float> {
  // parse - Return true on error.
  bool parse(Option &O, const char *AN, const std::string &Arg, float &Val);

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "number"; }
};


//--------------------------------------------------
// parser<std::string>
//
template<>
struct parser<std::string> : public basic_parser<std::string> {
  // parse - Return true on error.
  bool parse(Option &O, const char *ArgName, const std::string &Arg,
             std::string &Value) {
    Value = Arg;
    return false;
  }

  // getValueName - Overload in subclass to provide a better default value.
  virtual const char *getValueName() const { return "string"; }
};



//===----------------------------------------------------------------------===//
// applicator class - This class is used because we must use partial
// specialization to handle literal string arguments specially (const char* does
// not correctly respond to the apply method).  Because the syntax to use this
// is a pain, we have the 'apply' method below to handle the nastiness...
//
template<class Mod> struct applicator {
  template<class Opt>
  static void opt(const Mod &M, Opt &O) { M.apply(O); }
};

// Handle const char* as a special case...
template<unsigned n> struct applicator<char[n]> {
  template<class Opt>
  static void opt(const char *Str, Opt &O) { O.setArgStr(Str); }
};
template<unsigned n> struct applicator<const char[n]> {
  template<class Opt>
  static void opt(const char *Str, Opt &O) { O.setArgStr(Str); }
};
template<> struct applicator<const char*> {
  template<class Opt>
  static void opt(const char *Str, Opt &O) { O.setArgStr(Str); }
};

template<> struct applicator<NumOccurrences> {
  static void opt(NumOccurrences NO, Option &O) { O.setNumOccurrencesFlag(NO); }
};
template<> struct applicator<ValueExpected> {
  static void opt(ValueExpected VE, Option &O) { O.setValueExpectedFlag(VE); }
};
template<> struct applicator<OptionHidden> {
  static void opt(OptionHidden OH, Option &O) { O.setHiddenFlag(OH); }
};
template<> struct applicator<FormattingFlags> {
  static void opt(FormattingFlags FF, Option &O) { O.setFormattingFlag(FF); }
};
template<> struct applicator<MiscFlags> {
  static void opt(MiscFlags MF, Option &O) { O.setMiscFlag(MF); }
};

// apply method - Apply a modifier to an option in a type safe way.
template<class Mod, class Opt>
void apply(const Mod &M, Opt *O) {
  applicator<Mod>::opt(M, *O);
}


//===----------------------------------------------------------------------===//
// opt_storage class

// Default storage class definition: external storage.  This implementation
// assumes the user will specify a variable to store the data into with the
// cl::location(x) modifier.
//
template<class DataType, bool ExternalStorage, bool isClass>
class opt_storage {
  DataType *Location;   // Where to store the object...

  void check() {
    assert(Location != 0 && "cl::location(...) not specified for a command "
           "line option with external storage, "
           "or cl::init specified before cl::location()!!");
  }
public:
  opt_storage() : Location(0) {}

  bool setLocation(Option &O, DataType &L) {
    if (Location)
      return O.error(": cl::location(x) specified more than once!");
    Location = &L;
    return false;
  }

  template<class T>
  void setValue(const T &V) {
    check();
    *Location = V;
  }

  DataType &getValue() { check(); return *Location; }
  const DataType &getValue() const { check(); return *Location; }
};


// Define how to hold a class type object, such as a string.  Since we can
// inherit from a class, we do so.  This makes us exactly compatible with the
// object in all cases that it is used.
//
template<class DataType>
struct opt_storage<DataType,false,true> : public DataType {

  template<class T>
  void setValue(const T &V) { DataType::operator=(V); }

  DataType &getValue() { return *this; }
  const DataType &getValue() const { return *this; }
};

// Define a partial specialization to handle things we cannot inherit from.  In
// this case, we store an instance through containment, and overload operators
// to get at the value.
//
template<class DataType>
struct opt_storage<DataType, false, false> {
  DataType Value;

  // Make sure we initialize the value with the default constructor for the
  // type.
  opt_storage() : Value(DataType()) {}

  template<class T>
  void setValue(const T &V) { Value = V; }
  DataType &getValue() { return Value; }
  DataType getValue() const { return Value; }
};


//===----------------------------------------------------------------------===//
// opt - A scalar command line option.
//
template <class DataType, bool ExternalStorage = false,
          class ParserClass = parser<DataType> >
class opt : public Option, 
            public opt_storage<DataType, ExternalStorage,
                               is_class<DataType>::value> {
  ParserClass Parser;

  virtual bool handleOccurrence(const char *ArgName, const std::string &Arg) {
    typename ParserClass::parser_data_type Val;
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;                            // Parse error!
    setValue(Val);
    return false;
  }

  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return Parser.getValueExpectedFlagDefault();
  }

  // Forward printing stuff to the parser...
  virtual unsigned getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(unsigned GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

  void done() {
    addArgument(ArgStr);
    Parser.initialize(*this);
  }
public:
  // setInitialValue - Used by the cl::init modifier...
  void setInitialValue(const DataType &V) { this->setValue(V); }

  ParserClass &getParser() { return Parser; }

  operator DataType() const { return this->getValue(); }

  template<class T>
  DataType &operator=(const T &Val) {
    this->setValue(Val);
    return this->getValue();
  }

  // One option...
  template<class M0t>
  opt(const M0t &M0) {
    apply(M0, this);
    done();
  }

  // Two options...
  template<class M0t, class M1t>
  opt(const M0t &M0, const M1t &M1) {
    apply(M0, this); apply(M1, this);
    done();
  }

  // Three options...
  template<class M0t, class M1t, class M2t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5, const M6t &M6) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  opt(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5, const M6t &M6, const M7t &M7) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};

//===----------------------------------------------------------------------===//
// list_storage class

// Default storage class definition: external storage.  This implementation
// assumes the user will specify a variable to store the data into with the
// cl::location(x) modifier.
//
template<class DataType, class StorageClass>
class list_storage {
  StorageClass *Location;   // Where to store the object...

public:
  list_storage() : Location(0) {}

  bool setLocation(Option &O, StorageClass &L) {
    if (Location)
      return O.error(": cl::location(x) specified more than once!");
    Location = &L;
    return false;
  }

  template<class T>
  void addValue(const T &V) {
    assert(Location != 0 && "cl::location(...) not specified for a command "
           "line option with external storage!");
    Location->push_back(V);
  }
};


// Define how to hold a class type object, such as a string.  Since we can
// inherit from a class, we do so.  This makes us exactly compatible with the
// object in all cases that it is used.
//
template<class DataType>
struct list_storage<DataType, bool> : public std::vector<DataType> {

  template<class T>
  void addValue(const T &V) { push_back(V); }
};


//===----------------------------------------------------------------------===//
// list - A list of command line options.
//
template <class DataType, class Storage = bool,
          class ParserClass = parser<DataType> >
class list : public Option, public list_storage<DataType, Storage> {
  ParserClass Parser;

  virtual enum NumOccurrences getNumOccurrencesFlagDefault() const { 
    return ZeroOrMore;
  }
  virtual enum ValueExpected getValueExpectedFlagDefault() const {
    return Parser.getValueExpectedFlagDefault();
  }

  virtual bool handleOccurrence(const char *ArgName, const std::string &Arg) {
    typename ParserClass::parser_data_type Val;
    if (Parser.parse(*this, ArgName, Arg, Val))
      return true;  // Parse Error!
    addValue(Val);
    return false;
  }

  // Forward printing stuff to the parser...
  virtual unsigned getOptionWidth() const {return Parser.getOptionWidth(*this);}
  virtual void printOptionInfo(unsigned GlobalWidth) const {
    Parser.printOptionInfo(*this, GlobalWidth);
  }

  void done() {
    addArgument(ArgStr);
    Parser.initialize(*this);
  }
public:
  ParserClass &getParser() { return Parser; }

  // One option...
  template<class M0t>
  list(const M0t &M0) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  list(const M0t &M0, const M1t &M1) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  list(const M0t &M0, const M1t &M1, const M2t &M2) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
  // Five options...
  template<class M0t, class M1t, class M2t, class M3t, class M4t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this);
    done();
  }
  // Six options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
       const M4t &M4, const M5t &M5) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this);
    done();
  }
  // Seven options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5, const M6t &M6) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this);
    done();
  }
  // Eight options...
  template<class M0t, class M1t, class M2t, class M3t,
           class M4t, class M5t, class M6t, class M7t>
  list(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3,
      const M4t &M4, const M5t &M5, const M6t &M6, const M7t &M7) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    apply(M4, this); apply(M5, this); apply(M6, this); apply(M7, this);
    done();
  }
};



//===----------------------------------------------------------------------===//
// Aliased command line option (alias this name to a preexisting name)
//

class alias : public Option {
  Option *AliasFor;
  virtual bool handleOccurrence(const char *ArgName, const std::string &Arg) {
    return AliasFor->handleOccurrence(AliasFor->ArgStr, Arg);
  }
  // Aliases default to be hidden...
  virtual enum OptionHidden getOptionHiddenFlagDefault() const {return Hidden;}

  // Handle printing stuff...
  virtual unsigned getOptionWidth() const;
  virtual void printOptionInfo(unsigned GlobalWidth) const;

  void done() {
    if (!hasArgStr())
      error(": cl::alias must have argument name specified!");
    if (AliasFor == 0)
      error(": cl::alias must have an cl::aliasopt(option) specified!");
    addArgument(ArgStr);
  }
public:
  void setAliasFor(Option &O) {
    if (AliasFor)
      error(": cl::alias must only have one cl::aliasopt(...) specified!");
    AliasFor = &O;
  }

  // One option...
  template<class M0t>
  alias(const M0t &M0) : AliasFor(0) {
    apply(M0, this);
    done();
  }
  // Two options...
  template<class M0t, class M1t>
  alias(const M0t &M0, const M1t &M1) : AliasFor(0) {
    apply(M0, this); apply(M1, this);
    done();
  }
  // Three options...
  template<class M0t, class M1t, class M2t>
  alias(const M0t &M0, const M1t &M1, const M2t &M2) : AliasFor(0) {
    apply(M0, this); apply(M1, this); apply(M2, this);
    done();
  }
  // Four options...
  template<class M0t, class M1t, class M2t, class M3t>
  alias(const M0t &M0, const M1t &M1, const M2t &M2, const M3t &M3)
    : AliasFor(0) {
    apply(M0, this); apply(M1, this); apply(M2, this); apply(M3, this);
    done();
  }
};

// aliasfor - Modifier to set the option an alias aliases.
struct aliasopt {
  Option &Opt;
  aliasopt(Option &O) : Opt(O) {}
  void apply(alias &A) const { A.setAliasFor(Opt); }
};

} // End namespace cl

} // End namespace llvm

#endif
