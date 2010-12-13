//===- LLVMCConfigurationEmitter.cpp - Generate LLVMC config ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting LLVMC configuration code.
//
//===----------------------------------------------------------------------===//

#include "LLVMCConfigurationEmitter.h"
#include "Record.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>
#include <typeinfo>


using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
/// Typedefs

typedef std::vector<Record*> RecordVector;
typedef std::vector<const DagInit*> DagVector;
typedef std::vector<std::string> StrVector;

//===----------------------------------------------------------------------===//
/// Constants

// Indentation.
const unsigned TabWidth = 4;
const unsigned Indent1  = TabWidth*1;
const unsigned Indent2  = TabWidth*2;
const unsigned Indent3  = TabWidth*3;
const unsigned Indent4  = TabWidth*4;

// Default help string.
const char * const DefaultHelpString = "NO HELP MESSAGE PROVIDED";

// Name for the "sink" option.
const char * const SinkOptionName = "SinkOption";

//===----------------------------------------------------------------------===//
/// Helper functions

/// Id - An 'identity' function object.
struct Id {
  template<typename T0>
  void operator()(const T0&) const {
  }
  template<typename T0, typename T1>
  void operator()(const T0&, const T1&) const {
  }
  template<typename T0, typename T1, typename T2>
  void operator()(const T0&, const T1&, const T2&) const {
  }
};

int InitPtrToInt(const Init* ptr) {
  const IntInit& val = dynamic_cast<const IntInit&>(*ptr);
  return val.getValue();
}

const std::string& InitPtrToString(const Init* ptr) {
  const StringInit& val = dynamic_cast<const StringInit&>(*ptr);
  return val.getValue();
}

const ListInit& InitPtrToList(const Init* ptr) {
  const ListInit& val = dynamic_cast<const ListInit&>(*ptr);
  return val;
}

const DagInit& InitPtrToDag(const Init* ptr) {
  const DagInit& val = dynamic_cast<const DagInit&>(*ptr);
  return val;
}

const std::string GetOperatorName(const DagInit& D) {
  return D.getOperator()->getAsString();
}

/// CheckBooleanConstant - Check that the provided value is a boolean constant.
void CheckBooleanConstant(const Init* I) {
  const DefInit& val = dynamic_cast<const DefInit&>(*I);
  const std::string& str = val.getAsString();

  if (str != "true" && str != "false") {
    throw "Incorrect boolean value: '" + str +
      "': must be either 'true' or 'false'";
  }
}

// CheckNumberOfArguments - Ensure that the number of args in d is
// greater than or equal to min_arguments, otherwise throw an exception.
void CheckNumberOfArguments (const DagInit& d, unsigned minArgs) {
  if (d.getNumArgs() < minArgs)
    throw GetOperatorName(d) + ": too few arguments!";
}

// EscapeVariableName - Escape commas and other symbols not allowed
// in the C++ variable names. Makes it possible to use options named
// like "Wa," (useful for prefix options).
std::string EscapeVariableName (const std::string& Var) {
  std::string ret;
  for (unsigned i = 0; i != Var.size(); ++i) {
    char cur_char = Var[i];
    if (cur_char == ',') {
      ret += "_comma_";
    }
    else if (cur_char == '+') {
      ret += "_plus_";
    }
    else if (cur_char == '-') {
      ret += "_dash_";
    }
    else {
      ret.push_back(cur_char);
    }
  }
  return ret;
}

/// EscapeQuotes - Replace '"' with '\"'.
std::string EscapeQuotes (const std::string& Var) {
  std::string ret;
  for (unsigned i = 0; i != Var.size(); ++i) {
    char cur_char = Var[i];
    if (cur_char == '"') {
      ret += "\\\"";
    }
    else {
      ret.push_back(cur_char);
    }
  }
  return ret;
}

/// OneOf - Does the input string contain this character?
bool OneOf(const char* lst, char c) {
  while (*lst) {
    if (*lst++ == c)
      return true;
  }
  return false;
}

template <class I, class S>
void CheckedIncrement(I& P, I E, S ErrorString) {
  ++P;
  if (P == E)
    throw ErrorString;
}

//===----------------------------------------------------------------------===//
/// Back-end specific code


/// OptionType - One of six different option types. See the
/// documentation for detailed description of differences.
namespace OptionType {

  enum OptionType { Alias, Switch, SwitchList,
                    Parameter, ParameterList, Prefix, PrefixList };

  bool IsAlias(OptionType t) {
    return (t == Alias);
  }

  bool IsList (OptionType t) {
    return (t == SwitchList || t == ParameterList || t == PrefixList);
  }

  bool IsSwitch (OptionType t) {
    return (t == Switch);
  }

  bool IsSwitchList (OptionType t) {
    return (t == SwitchList);
  }

  bool IsParameter (OptionType t) {
    return (t == Parameter || t == Prefix);
  }

}

OptionType::OptionType stringToOptionType(const std::string& T) {
  if (T == "alias_option")
    return OptionType::Alias;
  else if (T == "switch_option")
    return OptionType::Switch;
  else if (T == "switch_list_option")
    return OptionType::SwitchList;
  else if (T == "parameter_option")
    return OptionType::Parameter;
  else if (T == "parameter_list_option")
    return OptionType::ParameterList;
  else if (T == "prefix_option")
    return OptionType::Prefix;
  else if (T == "prefix_list_option")
    return OptionType::PrefixList;
  else
    throw "Unknown option type: " + T + '!';
}

namespace OptionDescriptionFlags {
  enum OptionDescriptionFlags { Required = 0x1, Hidden = 0x2,
                                ReallyHidden = 0x4, OneOrMore = 0x8,
                                Optional = 0x10, CommaSeparated = 0x20,
                                ForwardNotSplit = 0x40, ZeroOrMore = 0x80 };
}

/// OptionDescription - Represents data contained in a single
/// OptionList entry.
struct OptionDescription {
  OptionType::OptionType Type;
  std::string Name;
  unsigned Flags;
  std::string Help;
  unsigned MultiVal;
  Init* InitVal;

  OptionDescription(OptionType::OptionType t = OptionType::Switch,
                    const std::string& n = "",
                    const std::string& h = DefaultHelpString)
    : Type(t), Name(n), Flags(0x0), Help(h), MultiVal(1), InitVal(0)
  {}

  /// GenTypeDeclaration - Returns the C++ variable type of this
  /// option.
  const char* GenTypeDeclaration() const;

  /// GenVariableName - Returns the variable name used in the
  /// generated C++ code.
  std::string GenVariableName() const
  { return "autogenerated::" + GenOptionType() + EscapeVariableName(Name); }

  /// GenPlainVariableName - Returns the variable name without the namespace
  /// prefix.
  std::string GenPlainVariableName() const
  { return GenOptionType() + EscapeVariableName(Name); }

  /// Merge - Merge two option descriptions.
  void Merge (const OptionDescription& other);

  /// CheckConsistency - Check that the flags are consistent.
  void CheckConsistency() const;

  // Misc convenient getters/setters.

  bool isAlias() const;

  bool isMultiVal() const;

  bool isCommaSeparated() const;
  void setCommaSeparated();

  bool isForwardNotSplit() const;
  void setForwardNotSplit();

  bool isRequired() const;
  void setRequired();

  bool isOneOrMore() const;
  void setOneOrMore();

  bool isZeroOrMore() const;
  void setZeroOrMore();

  bool isOptional() const;
  void setOptional();

  bool isHidden() const;
  void setHidden();

  bool isReallyHidden() const;
  void setReallyHidden();

  bool isSwitch() const
  { return OptionType::IsSwitch(this->Type); }

  bool isSwitchList() const
  { return OptionType::IsSwitchList(this->Type); }

  bool isParameter() const
  { return OptionType::IsParameter(this->Type); }

  bool isList() const
  { return OptionType::IsList(this->Type); }

  bool isParameterList() const
  { return (OptionType::IsList(this->Type)
            && !OptionType::IsSwitchList(this->Type)); }

private:

  // GenOptionType - Helper function used by GenVariableName().
  std::string GenOptionType() const;
};

void OptionDescription::CheckConsistency() const {
  unsigned i = 0;

  i += this->isRequired();
  i += this->isOptional();
  i += this->isOneOrMore();
  i += this->isZeroOrMore();

  if (i > 1) {
    throw "Only one of (required), (optional), (one_or_more) or "
      "(zero_or_more) properties is allowed!";
  }
}

void OptionDescription::Merge (const OptionDescription& other)
{
  if (other.Type != Type)
    throw "Conflicting definitions for the option " + Name + "!";

  if (Help == other.Help || Help == DefaultHelpString)
    Help = other.Help;
  else if (other.Help != DefaultHelpString) {
    llvm::errs() << "Warning: several different help strings"
      " defined for option " + Name + "\n";
  }

  Flags |= other.Flags;
}

bool OptionDescription::isAlias() const {
  return OptionType::IsAlias(this->Type);
}

bool OptionDescription::isMultiVal() const {
  return MultiVal > 1;
}

bool OptionDescription::isCommaSeparated() const {
  return Flags & OptionDescriptionFlags::CommaSeparated;
}
void OptionDescription::setCommaSeparated() {
  Flags |= OptionDescriptionFlags::CommaSeparated;
}

bool OptionDescription::isForwardNotSplit() const {
  return Flags & OptionDescriptionFlags::ForwardNotSplit;
}
void OptionDescription::setForwardNotSplit() {
  Flags |= OptionDescriptionFlags::ForwardNotSplit;
}

bool OptionDescription::isRequired() const {
  return Flags & OptionDescriptionFlags::Required;
}
void OptionDescription::setRequired() {
  Flags |= OptionDescriptionFlags::Required;
}

bool OptionDescription::isOneOrMore() const {
  return Flags & OptionDescriptionFlags::OneOrMore;
}
void OptionDescription::setOneOrMore() {
  Flags |= OptionDescriptionFlags::OneOrMore;
}

bool OptionDescription::isZeroOrMore() const {
  return Flags & OptionDescriptionFlags::ZeroOrMore;
}
void OptionDescription::setZeroOrMore() {
  Flags |= OptionDescriptionFlags::ZeroOrMore;
}

bool OptionDescription::isOptional() const {
  return Flags & OptionDescriptionFlags::Optional;
}
void OptionDescription::setOptional() {
  Flags |= OptionDescriptionFlags::Optional;
}

bool OptionDescription::isHidden() const {
  return Flags & OptionDescriptionFlags::Hidden;
}
void OptionDescription::setHidden() {
  Flags |= OptionDescriptionFlags::Hidden;
}

bool OptionDescription::isReallyHidden() const {
  return Flags & OptionDescriptionFlags::ReallyHidden;
}
void OptionDescription::setReallyHidden() {
  Flags |= OptionDescriptionFlags::ReallyHidden;
}

const char* OptionDescription::GenTypeDeclaration() const {
  switch (Type) {
  case OptionType::Alias:
    return "cl::alias";
  case OptionType::PrefixList:
  case OptionType::ParameterList:
    return "cl::list<std::string>";
  case OptionType::Switch:
    return "cl::opt<bool>";
  case OptionType::SwitchList:
    return "cl::list<bool>";
  case OptionType::Parameter:
  case OptionType::Prefix:
  default:
    return "cl::opt<std::string>";
  }
}

std::string OptionDescription::GenOptionType() const {
  switch (Type) {
  case OptionType::Alias:
    return "Alias_";
  case OptionType::PrefixList:
  case OptionType::ParameterList:
    return "List_";
  case OptionType::Switch:
    return "Switch_";
  case OptionType::SwitchList:
    return "SwitchList_";
  case OptionType::Prefix:
  case OptionType::Parameter:
  default:
    return "Parameter_";
  }
}

/// OptionDescriptions - An OptionDescription array plus some helper
/// functions.
class OptionDescriptions {
  typedef StringMap<OptionDescription> container_type;

  /// Descriptions - A list of OptionDescriptions.
  container_type Descriptions;

public:
  /// FindOption - exception-throwing wrapper for find().
  const OptionDescription& FindOption(const std::string& OptName) const;

  // Wrappers for FindOption that throw an exception in case the option has a
  // wrong type.
  const OptionDescription& FindSwitch(const std::string& OptName) const;
  const OptionDescription& FindParameter(const std::string& OptName) const;
  const OptionDescription& FindParameterList(const std::string& OptName) const;
  const OptionDescription&
  FindListOrParameter(const std::string& OptName) const;
  const OptionDescription&
  FindParameterListOrParameter(const std::string& OptName) const;

  /// insertDescription - Insert new OptionDescription into
  /// OptionDescriptions list
  void InsertDescription (const OptionDescription& o);

  // Support for STL-style iteration
  typedef container_type::const_iterator const_iterator;
  const_iterator begin() const { return Descriptions.begin(); }
  const_iterator end() const { return Descriptions.end(); }
};

const OptionDescription&
OptionDescriptions::FindOption(const std::string& OptName) const {
  const_iterator I = Descriptions.find(OptName);
  if (I != Descriptions.end())
    return I->second;
  else
    throw OptName + ": no such option!";
}

const OptionDescription&
OptionDescriptions::FindSwitch(const std::string& OptName) const {
  const OptionDescription& OptDesc = this->FindOption(OptName);
  if (!OptDesc.isSwitch())
    throw OptName + ": incorrect option type - should be a switch!";
  return OptDesc;
}

const OptionDescription&
OptionDescriptions::FindParameterList(const std::string& OptName) const {
  const OptionDescription& OptDesc = this->FindOption(OptName);
  if (!OptDesc.isList() || OptDesc.isSwitchList())
    throw OptName + ": incorrect option type - should be a parameter list!";
  return OptDesc;
}

const OptionDescription&
OptionDescriptions::FindParameter(const std::string& OptName) const {
  const OptionDescription& OptDesc = this->FindOption(OptName);
  if (!OptDesc.isParameter())
    throw OptName + ": incorrect option type - should be a parameter!";
  return OptDesc;
}

const OptionDescription&
OptionDescriptions::FindListOrParameter(const std::string& OptName) const {
  const OptionDescription& OptDesc = this->FindOption(OptName);
  if (!OptDesc.isList() && !OptDesc.isParameter())
    throw OptName
      + ": incorrect option type - should be a list or parameter!";
  return OptDesc;
}

const OptionDescription&
OptionDescriptions::FindParameterListOrParameter
(const std::string& OptName) const {
  const OptionDescription& OptDesc = this->FindOption(OptName);
  if ((!OptDesc.isList() && !OptDesc.isParameter()) || OptDesc.isSwitchList())
    throw OptName
      + ": incorrect option type - should be a parameter list or parameter!";
  return OptDesc;
}

void OptionDescriptions::InsertDescription (const OptionDescription& o) {
  container_type::iterator I = Descriptions.find(o.Name);
  if (I != Descriptions.end()) {
    OptionDescription& D = I->second;
    D.Merge(o);
  }
  else {
    Descriptions[o.Name] = o;
  }
}

/// HandlerTable - A base class for function objects implemented as
/// 'tables of handlers'.
template <typename Handler>
class HandlerTable {
protected:
  // Implementation details.

  /// HandlerMap - A map from property names to property handlers
  typedef StringMap<Handler> HandlerMap;

  static HandlerMap Handlers_;
  static bool staticMembersInitialized_;

public:

  Handler GetHandler (const std::string& HandlerName) const {
    typename HandlerMap::iterator method = Handlers_.find(HandlerName);

    if (method != Handlers_.end()) {
      Handler h = method->second;
      return h;
    }
    else {
      throw "No handler found for property " + HandlerName + "!";
    }
  }

  void AddHandler(const char* Property, Handler H) {
    Handlers_[Property] = H;
  }

};

template <class Handler, class FunctionObject>
Handler GetHandler(FunctionObject* Obj, const DagInit& Dag) {
  const std::string& HandlerName = GetOperatorName(Dag);
  return Obj->GetHandler(HandlerName);
}

template <class FunctionObject>
void InvokeDagInitHandler(FunctionObject* Obj, Init* I) {
  typedef void (FunctionObject::*Handler) (const DagInit&);

  const DagInit& Dag = InitPtrToDag(I);
  Handler h = GetHandler<Handler>(Obj, Dag);

  ((Obj)->*(h))(Dag);
}

template <class FunctionObject>
void InvokeDagInitHandler(const FunctionObject* const Obj,
                          const Init* I, unsigned IndentLevel, raw_ostream& O)
{
  typedef void (FunctionObject::*Handler)
    (const DagInit&, unsigned IndentLevel, raw_ostream& O) const;

  const DagInit& Dag = InitPtrToDag(I);
  Handler h = GetHandler<Handler>(Obj, Dag);

  ((Obj)->*(h))(Dag, IndentLevel, O);
}

template <typename H>
typename HandlerTable<H>::HandlerMap HandlerTable<H>::Handlers_;

template <typename H>
bool HandlerTable<H>::staticMembersInitialized_ = false;


/// CollectOptionProperties - Function object for iterating over an
/// option property list.
class CollectOptionProperties;
typedef void (CollectOptionProperties::* CollectOptionPropertiesHandler)
(const DagInit&);

class CollectOptionProperties
: public HandlerTable<CollectOptionPropertiesHandler>
{
private:

  /// optDescs_ - OptionDescriptions table. This is where the
  /// information is stored.
  OptionDescription& optDesc_;

public:

  explicit CollectOptionProperties(OptionDescription& OD)
    : optDesc_(OD)
  {
    if (!staticMembersInitialized_) {
      AddHandler("help", &CollectOptionProperties::onHelp);
      AddHandler("hidden", &CollectOptionProperties::onHidden);
      AddHandler("init", &CollectOptionProperties::onInit);
      AddHandler("multi_val", &CollectOptionProperties::onMultiVal);
      AddHandler("one_or_more", &CollectOptionProperties::onOneOrMore);
      AddHandler("zero_or_more", &CollectOptionProperties::onZeroOrMore);
      AddHandler("really_hidden", &CollectOptionProperties::onReallyHidden);
      AddHandler("required", &CollectOptionProperties::onRequired);
      AddHandler("optional", &CollectOptionProperties::onOptional);
      AddHandler("comma_separated", &CollectOptionProperties::onCommaSeparated);
      AddHandler("forward_not_split",
                 &CollectOptionProperties::onForwardNotSplit);

      staticMembersInitialized_ = true;
    }
  }

  /// operator() - Just forwards to the corresponding property
  /// handler.
  void operator() (Init* I) {
    InvokeDagInitHandler(this, I);
  }

private:

  /// Option property handlers --
  /// Methods that handle option properties such as (help) or (hidden).

  void onHelp (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    optDesc_.Help = EscapeQuotes(InitPtrToString(d.getArg(0)));
  }

  void onHidden (const DagInit& d) {
    CheckNumberOfArguments(d, 0);
    optDesc_.setHidden();
  }

  void onReallyHidden (const DagInit& d) {
    CheckNumberOfArguments(d, 0);
    optDesc_.setReallyHidden();
  }

  void onCommaSeparated (const DagInit& d) {
    CheckNumberOfArguments(d, 0);
    if (!optDesc_.isParameterList())
      throw "'comma_separated' is valid only on parameter list options!";
    optDesc_.setCommaSeparated();
  }

  void onForwardNotSplit (const DagInit& d) {
    CheckNumberOfArguments(d, 0);
    if (!optDesc_.isParameter())
      throw "'forward_not_split' is valid only for parameter options!";
    optDesc_.setForwardNotSplit();
  }

  void onRequired (const DagInit& d) {
    CheckNumberOfArguments(d, 0);

    optDesc_.setRequired();
    optDesc_.CheckConsistency();
  }

  void onInit (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    Init* i = d.getArg(0);
    const std::string& str = i->getAsString();

    bool correct = optDesc_.isParameter() && dynamic_cast<StringInit*>(i);
    correct |= (optDesc_.isSwitch() && (str == "true" || str == "false"));

    if (!correct)
      throw "Incorrect usage of the 'init' option property!";

    optDesc_.InitVal = i;
  }

  void onOneOrMore (const DagInit& d) {
    CheckNumberOfArguments(d, 0);

    optDesc_.setOneOrMore();
    optDesc_.CheckConsistency();
  }

  void onZeroOrMore (const DagInit& d) {
    CheckNumberOfArguments(d, 0);

    if (optDesc_.isList())
      llvm::errs() << "Warning: specifying the 'zero_or_more' property "
        "on a list option has no effect.\n";

    optDesc_.setZeroOrMore();
    optDesc_.CheckConsistency();
  }

  void onOptional (const DagInit& d) {
    CheckNumberOfArguments(d, 0);

    if (!optDesc_.isList())
      llvm::errs() << "Warning: specifying the 'optional' property"
        "on a non-list option has no effect.\n";

    optDesc_.setOptional();
    optDesc_.CheckConsistency();
  }

  void onMultiVal (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    int val = InitPtrToInt(d.getArg(0));
    if (val < 2)
      throw "Error in the 'multi_val' property: "
        "the value must be greater than 1!";
    if (!optDesc_.isParameterList())
      throw "The multi_val property is valid only on list options!";
    optDesc_.MultiVal = val;
  }

};

/// AddOption - A function object that is applied to every option
/// description. Used by CollectOptionDescriptions.
class AddOption {
private:
  OptionDescriptions& OptDescs_;

public:
  explicit AddOption(OptionDescriptions& OD) : OptDescs_(OD)
  {}

  void operator()(const Init* i) {
    const DagInit& d = InitPtrToDag(i);
    CheckNumberOfArguments(d, 1);

    const OptionType::OptionType Type =
      stringToOptionType(GetOperatorName(d));
    const std::string& Name = InitPtrToString(d.getArg(0));

    OptionDescription OD(Type, Name);

    CheckNumberOfArguments(d, 2);

    if (OD.isAlias()) {
      // Aliases store the aliased option name in the 'Help' field.
      OD.Help = InitPtrToString(d.getArg(1));
    }
    else {
      processOptionProperties(d, OD);
    }

    OptDescs_.InsertDescription(OD);
  }

private:
  /// processOptionProperties - Go through the list of option
  /// properties and call a corresponding handler for each.
  static void processOptionProperties (const DagInit& d, OptionDescription& o) {
    CheckNumberOfArguments(d, 2);
    DagInit::const_arg_iterator B = d.arg_begin();
    // Skip the first argument: it's always the option name.
    ++B;
    std::for_each(B, d.arg_end(), CollectOptionProperties(o));
  }

};

/// CollectOptionDescriptions - Collects option properties from all
/// OptionLists.
void CollectOptionDescriptions (const RecordVector& V,
                                OptionDescriptions& OptDescs)
{
  // For every OptionList:
  for (RecordVector::const_iterator B = V.begin(), E = V.end(); B!=E; ++B)
  {
    // Throws an exception if the value does not exist.
    ListInit* PropList = (*B)->getValueAsListInit("options");

    // For every option description in this list: invoke AddOption.
    std::for_each(PropList->begin(), PropList->end(), AddOption(OptDescs));
  }
}

// Tool information record

namespace ToolFlags {
  enum ToolFlags { Join = 0x1, Sink = 0x2 };
}

struct ToolDescription : public RefCountedBase<ToolDescription> {
  std::string Name;
  Init* CmdLine;
  Init* Actions;
  StrVector InLanguage;
  std::string InFileOption;
  std::string OutFileOption;
  StrVector OutLanguage;
  std::string OutputSuffix;
  unsigned Flags;
  const Init* OnEmpty;

  // Various boolean properties
  void setSink()      { Flags |= ToolFlags::Sink; }
  bool isSink() const { return Flags & ToolFlags::Sink; }
  void setJoin()      { Flags |= ToolFlags::Join; }
  bool isJoin() const { return Flags & ToolFlags::Join; }

  // Default ctor here is needed because StringMap can only store
  // DefaultConstructible objects
  ToolDescription (const std::string &n = "")
    : Name(n), CmdLine(0), Actions(0), OutFileOption("-o"),
      Flags(0), OnEmpty(0)
  {}
};

/// ToolDescriptions - A list of Tool information records.
typedef std::vector<IntrusiveRefCntPtr<ToolDescription> > ToolDescriptions;


/// CollectToolProperties - Function object for iterating over a list of
/// tool property records.

class CollectToolProperties;
typedef void (CollectToolProperties::* CollectToolPropertiesHandler)
(const DagInit&);

class CollectToolProperties : public HandlerTable<CollectToolPropertiesHandler>
{
private:

  /// toolDesc_ - Properties of the current Tool. This is where the
  /// information is stored.
  ToolDescription& toolDesc_;

public:

  explicit CollectToolProperties (ToolDescription& d)
    : toolDesc_(d)
  {
    if (!staticMembersInitialized_) {

      AddHandler("actions", &CollectToolProperties::onActions);
      AddHandler("command", &CollectToolProperties::onCommand);
      AddHandler("in_language", &CollectToolProperties::onInLanguage);
      AddHandler("join", &CollectToolProperties::onJoin);
      AddHandler("out_language", &CollectToolProperties::onOutLanguage);

      AddHandler("out_file_option", &CollectToolProperties::onOutFileOption);
      AddHandler("in_file_option", &CollectToolProperties::onInFileOption);

      AddHandler("output_suffix", &CollectToolProperties::onOutputSuffix);
      AddHandler("sink", &CollectToolProperties::onSink);
      AddHandler("works_on_empty", &CollectToolProperties::onWorksOnEmpty);

      staticMembersInitialized_ = true;
    }
  }

  void operator() (Init* I) {
    InvokeDagInitHandler(this, I);
  }

private:

  /// Property handlers --
  /// Functions that extract information about tool properties from
  /// DAG representation.

  void onActions (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    Init* Case = d.getArg(0);
    if (typeid(*Case) != typeid(DagInit) ||
        GetOperatorName(static_cast<DagInit&>(*Case)) != "case")
      throw "The argument to (actions) should be a 'case' construct!";
    toolDesc_.Actions = Case;
  }

  void onCommand (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    toolDesc_.CmdLine = d.getArg(0);
  }

  /// onInOutLanguage - Common implementation of on{In,Out}Language().
  void onInOutLanguage (const DagInit& d, StrVector& OutVec) {
    CheckNumberOfArguments(d, 1);

    // Copy strings to the output vector.
    for (unsigned i = 0, NumArgs = d.getNumArgs(); i < NumArgs; ++i) {
      OutVec.push_back(InitPtrToString(d.getArg(i)));
    }

    // Remove duplicates.
    std::sort(OutVec.begin(), OutVec.end());
    StrVector::iterator newE = std::unique(OutVec.begin(), OutVec.end());
    OutVec.erase(newE, OutVec.end());
  }


  void onInLanguage (const DagInit& d) {
    this->onInOutLanguage(d, toolDesc_.InLanguage);
  }

  void onJoin (const DagInit& d) {
    CheckNumberOfArguments(d, 0);
    toolDesc_.setJoin();
  }

  void onOutLanguage (const DagInit& d) {
    this->onInOutLanguage(d, toolDesc_.OutLanguage);
  }

  void onOutFileOption (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    toolDesc_.OutFileOption = InitPtrToString(d.getArg(0));
  }

  void onInFileOption (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    toolDesc_.InFileOption = InitPtrToString(d.getArg(0));
  }

  void onOutputSuffix (const DagInit& d) {
    CheckNumberOfArguments(d, 1);
    toolDesc_.OutputSuffix = InitPtrToString(d.getArg(0));
  }

  void onSink (const DagInit& d) {
    CheckNumberOfArguments(d, 0);
    toolDesc_.setSink();
  }

  void onWorksOnEmpty (const DagInit& d) {
    toolDesc_.OnEmpty = d.getArg(0);
  }

};

/// CollectToolDescriptions - Gather information about tool properties
/// from the parsed TableGen data (basically a wrapper for the
/// CollectToolProperties function object).
void CollectToolDescriptions (const RecordVector& Tools,
                              ToolDescriptions& ToolDescs)
{
  // Iterate over a properties list of every Tool definition
  for (RecordVector::const_iterator B = Tools.begin(),
         E = Tools.end(); B!=E; ++B) {
    const Record* T = *B;
    // Throws an exception if the value does not exist.
    ListInit* PropList = T->getValueAsListInit("properties");

    IntrusiveRefCntPtr<ToolDescription>
      ToolDesc(new ToolDescription(T->getName()));

    std::for_each(PropList->begin(), PropList->end(),
                  CollectToolProperties(*ToolDesc));
    ToolDescs.push_back(ToolDesc);
  }
}

/// FillInEdgeVector - Merge all compilation graph definitions into
/// one single edge list.
void FillInEdgeVector(const RecordVector& CompilationGraphs,
                      DagVector& Out) {
  for (RecordVector::const_iterator B = CompilationGraphs.begin(),
         E = CompilationGraphs.end(); B != E; ++B) {
    const ListInit* Edges = (*B)->getValueAsListInit("edges");

    for (ListInit::const_iterator B = Edges->begin(),
           E = Edges->end(); B != E; ++B) {
      Out.push_back(&InitPtrToDag(*B));
    }
  }
}

/// NotInGraph - Helper function object for FilterNotInGraph.
struct NotInGraph {
private:
  const llvm::StringSet<>& ToolsInGraph_;

public:
  NotInGraph(const llvm::StringSet<>& ToolsInGraph)
  : ToolsInGraph_(ToolsInGraph)
  {}

  bool operator()(const IntrusiveRefCntPtr<ToolDescription>& x) {
    return (ToolsInGraph_.count(x->Name) == 0);
  }
};

/// FilterNotInGraph - Filter out from ToolDescs all Tools not
/// mentioned in the compilation graph definition.
void FilterNotInGraph (const DagVector& EdgeVector,
                       ToolDescriptions& ToolDescs) {

  // List all tools mentioned in the graph.
  llvm::StringSet<> ToolsInGraph;

  for (DagVector::const_iterator B = EdgeVector.begin(),
         E = EdgeVector.end(); B != E; ++B) {

    const DagInit* Edge = *B;
    const std::string& NodeA = InitPtrToString(Edge->getArg(0));
    const std::string& NodeB = InitPtrToString(Edge->getArg(1));

    if (NodeA != "root")
      ToolsInGraph.insert(NodeA);
    ToolsInGraph.insert(NodeB);
  }

  // Filter ToolPropertiesList.
  ToolDescriptions::iterator new_end =
    std::remove_if(ToolDescs.begin(), ToolDescs.end(),
                   NotInGraph(ToolsInGraph));
  ToolDescs.erase(new_end, ToolDescs.end());
}

/// FillInToolToLang - Fills in two tables that map tool names to
/// input & output language names.  Helper function used by TypecheckGraph().
void FillInToolToLang (const ToolDescriptions& ToolDescs,
                       StringMap<StringSet<> >& ToolToInLang,
                       StringMap<StringSet<> >& ToolToOutLang) {
  for (ToolDescriptions::const_iterator B = ToolDescs.begin(),
         E = ToolDescs.end(); B != E; ++B) {
    const ToolDescription& D = *(*B);
    for (StrVector::const_iterator B = D.InLanguage.begin(),
           E = D.InLanguage.end(); B != E; ++B)
      ToolToInLang[D.Name].insert(*B);
    for (StrVector::const_iterator B = D.OutLanguage.begin(),
           E = D.OutLanguage.end(); B != E; ++B)
      ToolToOutLang[D.Name].insert(*B);
  }
}

/// Intersect - Is set intersection non-empty?
bool Intersect (const StringSet<>& S1, const StringSet<>& S2) {
  for (StringSet<>::const_iterator B = S1.begin(), E = S1.end(); B != E; ++B) {
    if (S2.count(B->first()) != 0)
      return true;
  }
  return false;
}

/// TypecheckGraph - Check that names for output and input languages
/// on all edges do match.
void TypecheckGraph (const DagVector& EdgeVector,
                     const ToolDescriptions& ToolDescs) {
  StringMap<StringSet<> > ToolToInLang;
  StringMap<StringSet<> > ToolToOutLang;

  FillInToolToLang(ToolDescs, ToolToInLang, ToolToOutLang);

  for (DagVector::const_iterator B = EdgeVector.begin(),
         E = EdgeVector.end(); B != E; ++B) {
    const DagInit* Edge = *B;
    const std::string& NodeA = InitPtrToString(Edge->getArg(0));
    const std::string& NodeB = InitPtrToString(Edge->getArg(1));
    StringMap<StringSet<> >::iterator IA = ToolToOutLang.find(NodeA);
    StringMap<StringSet<> >::iterator IB = ToolToInLang.find(NodeB);

    if (NodeB == "root")
      throw "Edges back to the root are not allowed!";

    if (NodeA != "root") {
      if (IA == ToolToOutLang.end())
        throw NodeA + ": no output language defined!";
      if (IB == ToolToInLang.end())
        throw NodeB + ": no input language defined!";

      if (!Intersect(IA->second, IB->second)) {
        throw "Edge " + NodeA + "->" + NodeB
          + ": output->input language mismatch";
      }
    }
  }
}

/// WalkCase - Walks the 'case' expression DAG and invokes
/// TestCallback on every test, and StatementCallback on every
/// statement. Handles 'case' nesting, but not the 'and' and 'or'
/// combinators (that is, they are passed directly to TestCallback).
/// TestCallback must have type 'void TestCallback(const DagInit*, unsigned
/// IndentLevel, bool FirstTest)'.
/// StatementCallback must have type 'void StatementCallback(const Init*,
/// unsigned IndentLevel)'.
template <typename F1, typename F2>
void WalkCase(const Init* Case, F1 TestCallback, F2 StatementCallback,
              unsigned IndentLevel = 0)
{
  const DagInit& d = InitPtrToDag(Case);

  // Error checks.
  if (GetOperatorName(d) != "case")
    throw "WalkCase should be invoked only on 'case' expressions!";

  if (d.getNumArgs() < 2)
    throw "There should be at least one clause in the 'case' expression:\n"
      + d.getAsString();

  // Main loop.
  bool even = false;
  const unsigned numArgs = d.getNumArgs();
  unsigned i = 1;
  for (DagInit::const_arg_iterator B = d.arg_begin(), E = d.arg_end();
       B != E; ++B) {
    Init* arg = *B;

    if (!even)
    {
      // Handle test.
      const DagInit& Test = InitPtrToDag(arg);

      if (GetOperatorName(Test) == "default" && (i+1 != numArgs))
        throw "The 'default' clause should be the last in the "
          "'case' construct!";
      if (i == numArgs)
        throw "Case construct handler: no corresponding action "
          "found for the test " + Test.getAsString() + '!';

      TestCallback(Test, IndentLevel, (i == 1));
    }
    else
    {
      if (dynamic_cast<DagInit*>(arg)
          && GetOperatorName(static_cast<DagInit&>(*arg)) == "case") {
        // Nested 'case'.
        WalkCase(arg, TestCallback, StatementCallback, IndentLevel + Indent1);
      }

      // Handle statement.
      StatementCallback(arg, IndentLevel);
    }

    ++i;
    even = !even;
  }
}

/// ExtractOptionNames - A helper function object used by
/// CheckForSuperfluousOptions() to walk the 'case' DAG.
class ExtractOptionNames {
  llvm::StringSet<>& OptionNames_;

  void processDag(const Init* Statement) {
    const DagInit& Stmt = InitPtrToDag(Statement);
    const std::string& ActionName = GetOperatorName(Stmt);
    if (ActionName == "forward" || ActionName == "forward_as" ||
        ActionName == "forward_value" ||
        ActionName == "forward_transformed_value" ||
        ActionName == "parameter_equals" || ActionName == "element_in_list") {
      CheckNumberOfArguments(Stmt, 1);

      Init* Arg = Stmt.getArg(0);
      if (typeid(*Arg) == typeid(StringInit))
        OptionNames_.insert(InitPtrToString(Arg));
    }
    else if (ActionName == "any_switch_on" || ActionName == "switch_on" ||
             ActionName == "any_not_empty" || ActionName == "any_empty" ||
             ActionName == "not_empty" || ActionName == "empty") {
      for (unsigned i = 0, NumArgs = Stmt.getNumArgs(); i < NumArgs; ++i) {
        Init* Arg = Stmt.getArg(i);
        if (typeid(*Arg) == typeid(StringInit))
          OptionNames_.insert(InitPtrToString(Arg));
      }
    }
    else if (ActionName == "and" || ActionName == "or" || ActionName == "not") {
      for (unsigned i = 0, NumArgs = Stmt.getNumArgs(); i < NumArgs; ++i) {
        this->processDag(Stmt.getArg(i));
      }
    }
  }

public:
  ExtractOptionNames(llvm::StringSet<>& OptionNames) : OptionNames_(OptionNames)
  {}

  void operator()(const Init* Statement) {
    // Statement is either a dag, or a list of dags.
    if (typeid(*Statement) == typeid(ListInit)) {
      const ListInit& DagList = *static_cast<const ListInit*>(Statement);
      for (ListInit::const_iterator B = DagList.begin(), E = DagList.end();
           B != E; ++B)
        this->processDag(*B);
    }
    else {
      this->processDag(Statement);
    }
  }

  void operator()(const DagInit& Test, unsigned, bool) {
    this->operator()(&Test);
  }
  void operator()(const Init* Statement, unsigned) {
    this->operator()(Statement);
  }
};

/// IsOptionalEdge - Validate that the 'optional_edge' has proper structure.
bool IsOptionalEdge (const DagInit& Edg) {
  return (GetOperatorName(Edg) == "optional_edge") && (Edg.getNumArgs() > 2);
}

/// CheckForSuperfluousOptions - Check that there are no side
/// effect-free options (specified only in the OptionList). Otherwise,
/// output a warning.
void CheckForSuperfluousOptions (const DagVector& EdgeVector,
                                 const ToolDescriptions& ToolDescs,
                                 const OptionDescriptions& OptDescs) {
  llvm::StringSet<> nonSuperfluousOptions;

  // Add all options mentioned in the ToolDesc.Actions to the set of
  // non-superfluous options.
  for (ToolDescriptions::const_iterator B = ToolDescs.begin(),
         E = ToolDescs.end(); B != E; ++B) {
    const ToolDescription& TD = *(*B);
    ExtractOptionNames Callback(nonSuperfluousOptions);
    if (TD.Actions)
      WalkCase(TD.Actions, Callback, Callback);
  }

  // Add all options mentioned in the 'case' clauses of the
  // OptionalEdges of the compilation graph to the set of
  // non-superfluous options.
  for (DagVector::const_iterator B = EdgeVector.begin(),
         E = EdgeVector.end(); B != E; ++B) {
    const DagInit& Edge = **B;
    if (IsOptionalEdge(Edge)) {
      const DagInit& Weight = InitPtrToDag(Edge.getArg(2));
      WalkCase(&Weight, ExtractOptionNames(nonSuperfluousOptions), Id());
    }
  }

  // Check that all options in OptDescs belong to the set of
  // non-superfluous options.
  for (OptionDescriptions::const_iterator B = OptDescs.begin(),
         E = OptDescs.end(); B != E; ++B) {
    const OptionDescription& Val = B->second;
    if (!nonSuperfluousOptions.count(Val.Name)
        && Val.Type != OptionType::Alias)
      llvm::errs() << "Warning: option '-" << Val.Name << "' has no effect! "
        "Probable cause: this option is specified only in the OptionList.\n";
  }
}

/// EmitCaseTest0Args - Helper function used by EmitCaseConstructHandler().
bool EmitCaseTest0Args(const std::string& TestName, raw_ostream& O) {
  if (TestName == "single_input_file") {
    O << "InputFilenames.size() == 1";
    return true;
  }
  else if (TestName == "multiple_input_files") {
    O << "InputFilenames.size() > 1";
    return true;
  }

  return false;
}

/// EmitMultipleArgumentTest - Helper function used by
/// EmitCaseTestMultipleArgs()
template <typename F>
void EmitMultipleArgumentTest(const DagInit& D, const char* LogicOp,
                              F Callback, raw_ostream& O)
{
  for (unsigned i = 0, NumArgs = D.getNumArgs(); i < NumArgs; ++i) {
    if (i != 0)
       O << ' ' << LogicOp << ' ';
    Callback(InitPtrToString(D.getArg(i)), O);
  }
}

// Callbacks for use with EmitMultipleArgumentTest

class EmitSwitchOn {
  const OptionDescriptions& OptDescs_;
public:
  EmitSwitchOn(const OptionDescriptions& OptDescs) : OptDescs_(OptDescs)
  {}

  void operator()(const std::string& OptName, raw_ostream& O) const {
    const OptionDescription& OptDesc = OptDescs_.FindSwitch(OptName);
    O << OptDesc.GenVariableName();
  }
};

class EmitEmptyTest {
  bool EmitNegate_;
  const OptionDescriptions& OptDescs_;
public:
  EmitEmptyTest(bool EmitNegate, const OptionDescriptions& OptDescs)
    : EmitNegate_(EmitNegate), OptDescs_(OptDescs)
  {}

  void operator()(const std::string& OptName, raw_ostream& O) const {
    const char* Neg = (EmitNegate_ ? "!" : "");
    if (OptName == "o") {
      O << Neg << "OutputFilename.empty()";
    }
    else if (OptName == "save-temps") {
      O << Neg << "(SaveTemps == SaveTempsEnum::Unset)";
    }
    else {
      const OptionDescription& OptDesc = OptDescs_.FindListOrParameter(OptName);
      O << Neg << OptDesc.GenVariableName() << ".empty()";
    }
  }
};


/// EmitCaseTestMultipleArgs - Helper function used by EmitCaseTest1Arg()
bool EmitCaseTestMultipleArgs (const std::string& TestName,
                               const DagInit& d,
                               const OptionDescriptions& OptDescs,
                               raw_ostream& O) {
  if (TestName == "any_switch_on") {
    EmitMultipleArgumentTest(d, "||", EmitSwitchOn(OptDescs), O);
    return true;
  }
  else if (TestName == "switch_on") {
    EmitMultipleArgumentTest(d, "&&", EmitSwitchOn(OptDescs), O);
    return true;
  }
  else if (TestName == "any_not_empty") {
    EmitMultipleArgumentTest(d, "||", EmitEmptyTest(true, OptDescs), O);
    return true;
  }
  else if (TestName == "any_empty") {
    EmitMultipleArgumentTest(d, "||", EmitEmptyTest(false, OptDescs), O);
    return true;
  }
  else if (TestName == "not_empty") {
    EmitMultipleArgumentTest(d, "&&", EmitEmptyTest(true, OptDescs), O);
    return true;
  }
  else if (TestName == "empty") {
    EmitMultipleArgumentTest(d, "&&", EmitEmptyTest(false, OptDescs), O);
    return true;
  }

  return false;
}

/// EmitCaseTest1Arg - Helper function used by EmitCaseTest1OrMoreArgs()
bool EmitCaseTest1Arg (const std::string& TestName,
                       const DagInit& d,
                       const OptionDescriptions& OptDescs,
                       raw_ostream& O) {
  const std::string& Arg = InitPtrToString(d.getArg(0));

  if (TestName == "input_languages_contain") {
    O << "InLangs.count(\"" << Arg << "\") != 0";
    return true;
  }
  else if (TestName == "in_language") {
    // This works only for single-argument Tool::GenerateAction. Join
    // tools can process several files in different languages simultaneously.

    // TODO: make this work with Edge::Weight (if possible).
    O << "LangMap.GetLanguage(inFile) == \"" << Arg << '\"';
    return true;
  }

  return false;
}

/// EmitCaseTest1OrMoreArgs - Helper function used by
/// EmitCaseConstructHandler()
bool EmitCaseTest1OrMoreArgs(const std::string& TestName,
                             const DagInit& d,
                             const OptionDescriptions& OptDescs,
                             raw_ostream& O) {
  CheckNumberOfArguments(d, 1);
  return EmitCaseTest1Arg(TestName, d, OptDescs, O) ||
    EmitCaseTestMultipleArgs(TestName, d, OptDescs, O);
}

/// EmitCaseTest2Args - Helper function used by EmitCaseConstructHandler().
bool EmitCaseTest2Args(const std::string& TestName,
                       const DagInit& d,
                       unsigned IndentLevel,
                       const OptionDescriptions& OptDescs,
                       raw_ostream& O) {
  CheckNumberOfArguments(d, 2);
  const std::string& OptName = InitPtrToString(d.getArg(0));
  const std::string& OptArg = InitPtrToString(d.getArg(1));

  if (TestName == "parameter_equals") {
    const OptionDescription& OptDesc = OptDescs.FindParameter(OptName);
    O << OptDesc.GenVariableName() << " == \"" << OptArg << "\"";
    return true;
  }
  else if (TestName == "element_in_list") {
    const OptionDescription& OptDesc = OptDescs.FindParameterList(OptName);
    const std::string& VarName = OptDesc.GenVariableName();
    O << "std::find(" << VarName << ".begin(),\n";
    O.indent(IndentLevel + Indent1)
      << VarName << ".end(), \""
      << OptArg << "\") != " << VarName << ".end()";
    return true;
  }

  return false;
}

// Forward declaration.
// EmitLogicalOperationTest and EmitCaseTest are mutually recursive.
void EmitCaseTest(const DagInit& d, unsigned IndentLevel,
                  const OptionDescriptions& OptDescs,
                  raw_ostream& O);

/// EmitLogicalOperationTest - Helper function used by
/// EmitCaseConstructHandler.
void EmitLogicalOperationTest(const DagInit& d, const char* LogicOp,
                              unsigned IndentLevel,
                              const OptionDescriptions& OptDescs,
                              raw_ostream& O) {
  O << '(';
  for (unsigned i = 0, NumArgs = d.getNumArgs(); i < NumArgs; ++i) {
    const DagInit& InnerTest = InitPtrToDag(d.getArg(i));
    EmitCaseTest(InnerTest, IndentLevel, OptDescs, O);
    if (i != NumArgs - 1) {
      O << ")\n";
      O.indent(IndentLevel + Indent1) << ' ' << LogicOp << " (";
    }
    else {
      O << ')';
    }
  }
}

void EmitLogicalNot(const DagInit& d, unsigned IndentLevel,
                    const OptionDescriptions& OptDescs, raw_ostream& O)
{
  CheckNumberOfArguments(d, 1);
  const DagInit& InnerTest = InitPtrToDag(d.getArg(0));
  O << "! (";
  EmitCaseTest(InnerTest, IndentLevel, OptDescs, O);
  O << ")";
}

/// EmitCaseTest - Helper function used by EmitCaseConstructHandler.
void EmitCaseTest(const DagInit& d, unsigned IndentLevel,
                  const OptionDescriptions& OptDescs,
                  raw_ostream& O) {
  const std::string& TestName = GetOperatorName(d);

  if (TestName == "and")
    EmitLogicalOperationTest(d, "&&", IndentLevel, OptDescs, O);
  else if (TestName == "or")
    EmitLogicalOperationTest(d, "||", IndentLevel, OptDescs, O);
  else if (TestName == "not")
    EmitLogicalNot(d, IndentLevel, OptDescs, O);
  else if (EmitCaseTest0Args(TestName, O))
    return;
  else if (EmitCaseTest1OrMoreArgs(TestName, d, OptDescs, O))
    return;
  else if (EmitCaseTest2Args(TestName, d, IndentLevel, OptDescs, O))
    return;
  else
    throw "Unknown test '" + TestName + "' used in the 'case' construct!";
}


/// EmitCaseTestCallback - Callback used by EmitCaseConstructHandler.
class EmitCaseTestCallback {
  bool EmitElseIf_;
  const OptionDescriptions& OptDescs_;
  raw_ostream& O_;
public:

  EmitCaseTestCallback(bool EmitElseIf,
                       const OptionDescriptions& OptDescs, raw_ostream& O)
    : EmitElseIf_(EmitElseIf), OptDescs_(OptDescs), O_(O)
  {}

  void operator()(const DagInit& Test, unsigned IndentLevel, bool FirstTest)
  {
    if (GetOperatorName(Test) == "default") {
      O_.indent(IndentLevel) << "else {\n";
    }
    else {
      O_.indent(IndentLevel)
        << ((!FirstTest && EmitElseIf_) ? "else if (" : "if (");
      EmitCaseTest(Test, IndentLevel, OptDescs_, O_);
      O_ << ") {\n";
    }
  }
};

/// EmitCaseStatementCallback - Callback used by EmitCaseConstructHandler.
template <typename F>
class EmitCaseStatementCallback {
  F Callback_;
  raw_ostream& O_;
public:

  EmitCaseStatementCallback(F Callback, raw_ostream& O)
    : Callback_(Callback), O_(O)
  {}

  void operator() (const Init* Statement, unsigned IndentLevel) {
    // Is this a nested 'case'?
    bool IsCase = dynamic_cast<const DagInit*>(Statement) &&
      GetOperatorName(static_cast<const DagInit&>(*Statement)) == "case";

    // If so, ignore it, it is handled by our caller, WalkCase.
    if (!IsCase) {
      if (typeid(*Statement) == typeid(ListInit)) {
        const ListInit& DagList = *static_cast<const ListInit*>(Statement);
        for (ListInit::const_iterator B = DagList.begin(), E = DagList.end();
             B != E; ++B)
          Callback_(*B, (IndentLevel + Indent1), O_);
      }
      else {
        Callback_(Statement, (IndentLevel + Indent1), O_);
      }
    }
    O_.indent(IndentLevel) << "}\n";
  }

};

/// EmitCaseConstructHandler - Emit code that handles the 'case'
/// construct. Takes a function object that should emit code for every case
/// clause. Implemented on top of WalkCase.
/// Callback's type is void F(const Init* Statement, unsigned IndentLevel,
/// raw_ostream& O).
/// EmitElseIf parameter controls the type of condition that is emitted ('if
/// (..) {..} else if (..) {} .. else {..}' vs. 'if (..) {..} if(..)  {..}
/// .. else {..}').
template <typename F>
void EmitCaseConstructHandler(const Init* Case, unsigned IndentLevel,
                              F Callback, bool EmitElseIf,
                              const OptionDescriptions& OptDescs,
                              raw_ostream& O) {
  WalkCase(Case, EmitCaseTestCallback(EmitElseIf, OptDescs, O),
           EmitCaseStatementCallback<F>(Callback, O), IndentLevel);
}

/// TokenizeCmdLine - converts from
/// "$CALL(HookName, 'Arg1', 'Arg2')/path -arg1 -arg2" to
/// ["$CALL(", "HookName", "Arg1", "Arg2", ")/path", "-arg1", "-arg2"].
void TokenizeCmdLine(const std::string& CmdLine, StrVector& Out) {
  const char* Delimiters = " \t\n\v\f\r";
  enum TokenizerState
  { Normal, SpecialCommand, InsideSpecialCommand, InsideQuotationMarks }
  cur_st  = Normal;

  if (CmdLine.empty())
    return;
  Out.push_back("");

  std::string::size_type B = CmdLine.find_first_not_of(Delimiters),
    E = CmdLine.size();

  for (; B != E; ++B) {
    char cur_ch = CmdLine[B];

    switch (cur_st) {
    case Normal:
      if (cur_ch == '$') {
        cur_st = SpecialCommand;
        break;
      }
      if (OneOf(Delimiters, cur_ch)) {
        // Skip whitespace
        B = CmdLine.find_first_not_of(Delimiters, B);
        if (B == std::string::npos) {
          B = E-1;
          continue;
        }
        --B;
        Out.push_back("");
        continue;
      }
      break;


    case SpecialCommand:
      if (OneOf(Delimiters, cur_ch)) {
        cur_st = Normal;
        Out.push_back("");
        continue;
      }
      if (cur_ch == '(') {
        Out.push_back("");
        cur_st = InsideSpecialCommand;
        continue;
      }
      break;

    case InsideSpecialCommand:
      if (OneOf(Delimiters, cur_ch)) {
        continue;
      }
      if (cur_ch == '\'') {
        cur_st = InsideQuotationMarks;
        Out.push_back("");
        continue;
      }
      if (cur_ch == ')') {
        cur_st = Normal;
        Out.push_back("");
      }
      if (cur_ch == ',') {
        continue;
      }

      break;

    case InsideQuotationMarks:
      if (cur_ch == '\'') {
        cur_st = InsideSpecialCommand;
        continue;
      }
      break;
    }

    Out.back().push_back(cur_ch);
  }
}

/// SubstituteCall - Given "$CALL(HookName, [Arg1 [, Arg2 [...]]])", output
/// "hooks::HookName([Arg1 [, Arg2 [, ...]]])". Helper function used by
/// SubstituteSpecialCommands().
StrVector::const_iterator
SubstituteCall (StrVector::const_iterator Pos,
                StrVector::const_iterator End,
                bool IsJoin, raw_ostream& O)
{
  const char* errorMessage = "Syntax error in $CALL invocation!";
  CheckedIncrement(Pos, End, errorMessage);
  const std::string& CmdName = *Pos;

  if (CmdName == ")")
    throw "$CALL invocation: empty argument list!";

  O << "hooks::";
  O << CmdName << "(";


  bool firstIteration = true;
  while (true) {
    CheckedIncrement(Pos, End, errorMessage);
    const std::string& Arg = *Pos;
    assert(Arg.size() != 0);

    if (Arg[0] == ')')
      break;

    if (firstIteration)
      firstIteration = false;
    else
      O << ", ";

    if (Arg == "$INFILE") {
      if (IsJoin)
        throw "$CALL(Hook, $INFILE) can't be used with a Join tool!";
      else
        O << "inFile.c_str()";
    }
    else {
      O << '"' << Arg << '"';
    }
  }

  O << ')';

  return Pos;
}

/// SubstituteEnv - Given '$ENV(VAR_NAME)', output 'getenv("VAR_NAME")'. Helper
/// function used by SubstituteSpecialCommands().
StrVector::const_iterator
SubstituteEnv (StrVector::const_iterator Pos,
               StrVector::const_iterator End, raw_ostream& O)
{
  const char* errorMessage = "Syntax error in $ENV invocation!";
  CheckedIncrement(Pos, End, errorMessage);
  const std::string& EnvName = *Pos;

  if (EnvName == ")")
    throw "$ENV invocation: empty argument list!";

  O << "checkCString(std::getenv(\"";
  O << EnvName;
  O << "\"))";

  CheckedIncrement(Pos, End, errorMessage);

  return Pos;
}

/// SubstituteSpecialCommands - Given an invocation of $CALL or $ENV, output
/// handler code. Helper function used by EmitCmdLineVecFill().
StrVector::const_iterator
SubstituteSpecialCommands (StrVector::const_iterator Pos,
                           StrVector::const_iterator End,
                           bool IsJoin, raw_ostream& O)
{

  const std::string& cmd = *Pos;

  // Perform substitution.
  if (cmd == "$CALL") {
    Pos = SubstituteCall(Pos, End, IsJoin, O);
  }
  else if (cmd == "$ENV") {
    Pos = SubstituteEnv(Pos, End, O);
  }
  else {
    throw "Unknown special command: " + cmd;
  }

  // Handle '$CMD(ARG)/additional/text'.
  const std::string& Leftover = *Pos;
  assert(Leftover.at(0) == ')');
  if (Leftover.size() != 1)
    O << " + std::string(\"" << (Leftover.c_str() + 1) << "\")";

  return Pos;
}

/// EmitCmdLineVecFill - Emit code that fills in the command line
/// vector. Helper function used by EmitGenerateActionMethod().
void EmitCmdLineVecFill(const Init* CmdLine, const std::string& ToolName,
                        bool IsJoin, unsigned IndentLevel,
                        raw_ostream& O) {
  StrVector StrVec;
  TokenizeCmdLine(InitPtrToString(CmdLine), StrVec);

  if (StrVec.empty())
    throw "Tool '" + ToolName + "' has empty command line!";

  StrVector::const_iterator B = StrVec.begin(), E = StrVec.end();

  // Emit the command itself.
  assert(!StrVec[0].empty());
  O.indent(IndentLevel) << "cmd = ";
  if (StrVec[0][0] == '$') {
    B = SubstituteSpecialCommands(B, E, IsJoin, O);
    ++B;
  }
  else {
    O << '"' << StrVec[0] << '"';
    ++B;
  }
  O << ";\n";

  // Go through the command arguments.
  assert(B <= E);
  for (; B != E; ++B) {
    const std::string& cmd = *B;

    assert(!cmd.empty());
    O.indent(IndentLevel);

    if (cmd.at(0) == '$') {
      O << "vec.push_back(std::make_pair(0, ";
      B = SubstituteSpecialCommands(B, E, IsJoin, O);
      O << "));\n";
    }
    else {
      O << "vec.push_back(std::make_pair(0, \"" << cmd << "\"));\n";
    }
  }

}

/// EmitForEachListElementCycleHeader - Emit common code for iterating through
/// all elements of a list. Helper function used by
/// EmitForwardOptionPropertyHandlingCode.
void EmitForEachListElementCycleHeader (const OptionDescription& D,
                                        unsigned IndentLevel,
                                        raw_ostream& O) {
  unsigned IndentLevel1 = IndentLevel + Indent1;

  O.indent(IndentLevel)
    << "for (" << D.GenTypeDeclaration()
    << "::iterator B = " << D.GenVariableName() << ".begin(),\n";
  O.indent(IndentLevel)
    << "E = " << D.GenVariableName() << ".end(); B != E;) {\n";
  O.indent(IndentLevel1) << "unsigned pos = " << D.GenVariableName()
                         << ".getPosition(B - " << D.GenVariableName()
                         << ".begin());\n";
}

/// EmitForwardOptionPropertyHandlingCode - Helper function used to
/// implement EmitActionHandler. Emits code for
/// handling the (forward) and (forward_as) option properties.
void EmitForwardOptionPropertyHandlingCode (const OptionDescription& D,
                                            unsigned IndentLevel,
                                            const std::string& NewName,
                                            raw_ostream& O) {
  const std::string& Name = NewName.empty()
    ? ("-" + D.Name)
    : NewName;
  unsigned IndentLevel1 = IndentLevel + Indent1;

  switch (D.Type) {
  case OptionType::Switch:
    O.indent(IndentLevel)
      << "vec.push_back(std::make_pair(" << D.GenVariableName()
      << ".getPosition(), \"" << Name << "\"));\n";
    break;
  case OptionType::Parameter:
    O.indent(IndentLevel) << "vec.push_back(std::make_pair("
                          << D.GenVariableName()
                          <<".getPosition(), \"" << Name;

    if (!D.isForwardNotSplit()) {
      O << "\"));\n";
      O.indent(IndentLevel) << "vec.push_back(std::make_pair("
                            << D.GenVariableName() << ".getPosition(), "
                            << D.GenVariableName() << "));\n";
    }
    else {
      O << "=\" + " << D.GenVariableName() << "));\n";
    }
    break;
  case OptionType::Prefix:
    O.indent(IndentLevel) << "vec.push_back(std::make_pair("
                          << D.GenVariableName() << ".getPosition(), \""
                          << Name << "\" + "
                          << D.GenVariableName() << "));\n";
    break;
  case OptionType::PrefixList:
    EmitForEachListElementCycleHeader(D, IndentLevel, O);
    O.indent(IndentLevel1) << "vec.push_back(std::make_pair(pos, \""
                           << Name << "\" + " << "*B));\n";
    O.indent(IndentLevel1) << "++B;\n";

    for (int i = 1, j = D.MultiVal; i < j; ++i) {
      O.indent(IndentLevel1) << "vec.push_back(std::make_pair(pos, *B));\n";
      O.indent(IndentLevel1) << "++B;\n";
    }

    O.indent(IndentLevel) << "}\n";
    break;
  case OptionType::ParameterList:
    EmitForEachListElementCycleHeader(D, IndentLevel, O);
    O.indent(IndentLevel1) << "vec.push_back(std::make_pair(pos, \""
                           << Name << "\"));\n";

    for (int i = 0, j = D.MultiVal; i < j; ++i) {
      O.indent(IndentLevel1) << "vec.push_back(std::make_pair(pos, *B));\n";
      O.indent(IndentLevel1) << "++B;\n";
    }

    O.indent(IndentLevel) << "}\n";
    break;
  case OptionType::SwitchList:
    EmitForEachListElementCycleHeader(D, IndentLevel, O);
    O.indent(IndentLevel1) << "vec.push_back(std::make_pair(pos, \""
                           << Name << "\"));\n";
    O.indent(IndentLevel1) << "++B;\n";
    O.indent(IndentLevel) << "}\n";
    break;
  case OptionType::Alias:
  default:
    throw "Aliases are not allowed in tool option descriptions!";
  }
}

/// ActionHandlingCallbackBase - Base class of EmitActionHandlersCallback and
/// EmitPreprocessOptionsCallback.
struct ActionHandlingCallbackBase
{

  void onErrorDag(const DagInit& d,
                  unsigned IndentLevel, raw_ostream& O) const
  {
    O.indent(IndentLevel)
      << "PrintError(\""
      << (d.getNumArgs() >= 1 ? InitPtrToString(d.getArg(0)) : "Unknown error!")
      << "\");\n";
    O.indent(IndentLevel) << "return 1;\n";
  }

  void onWarningDag(const DagInit& d,
                    unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(d, 1);
    O.indent(IndentLevel) << "llvm::errs() << \""
                          << InitPtrToString(d.getArg(0)) << "\";\n";
  }

};

/// EmitActionHandlersCallback - Emit code that handles actions. Used by
/// EmitGenerateActionMethod() as an argument to EmitCaseConstructHandler().
class EmitActionHandlersCallback;

typedef void (EmitActionHandlersCallback::* EmitActionHandlersCallbackHandler)
(const DagInit&, unsigned, raw_ostream&) const;

class EmitActionHandlersCallback :
  public ActionHandlingCallbackBase,
  public HandlerTable<EmitActionHandlersCallbackHandler>
{
  typedef EmitActionHandlersCallbackHandler Handler;

  const OptionDescriptions& OptDescs;

  /// EmitHookInvocation - Common code for hook invocation from actions. Used by
  /// onAppendCmd and onOutputSuffix.
  void EmitHookInvocation(const std::string& Str,
                          const char* BlockOpen, const char* BlockClose,
                          unsigned IndentLevel, raw_ostream& O) const
  {
    StrVector Out;
    TokenizeCmdLine(Str, Out);

    for (StrVector::const_iterator B = Out.begin(), E = Out.end();
         B != E; ++B) {
      const std::string& cmd = *B;

      O.indent(IndentLevel) << BlockOpen;

      if (cmd.at(0) == '$')
        B = SubstituteSpecialCommands(B, E,  /* IsJoin = */ true, O);
      else
        O << '"' << cmd << '"';

      O << BlockClose;
    }
  }

  void onAppendCmd (const DagInit& Dag,
                    unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 1);
    this->EmitHookInvocation(InitPtrToString(Dag.getArg(0)),
                             "vec.push_back(std::make_pair(65536, ", "));\n",
                             IndentLevel, O);
  }

  void onForward (const DagInit& Dag,
                  unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 1);
    const std::string& Name = InitPtrToString(Dag.getArg(0));
    EmitForwardOptionPropertyHandlingCode(OptDescs.FindOption(Name),
                                          IndentLevel, "", O);
  }

  void onForwardAs (const DagInit& Dag,
                    unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 2);
    const std::string& Name = InitPtrToString(Dag.getArg(0));
    const std::string& NewName = InitPtrToString(Dag.getArg(1));
    EmitForwardOptionPropertyHandlingCode(OptDescs.FindOption(Name),
                                          IndentLevel, NewName, O);
  }

  void onForwardValue (const DagInit& Dag,
                       unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 1);
    const std::string& Name = InitPtrToString(Dag.getArg(0));
    const OptionDescription& D = OptDescs.FindParameterListOrParameter(Name);

    if (D.isSwitchList()) {
      throw std::runtime_error
        ("forward_value is not allowed with switch_list");
    }

    if (D.isParameter()) {
      O.indent(IndentLevel) << "vec.push_back(std::make_pair("
                            << D.GenVariableName() << ".getPosition(), "
                            << D.GenVariableName() << "));\n";
    }
    else {
      O.indent(IndentLevel) << "for (" << D.GenTypeDeclaration()
                            << "::iterator B = " << D.GenVariableName()
                            << ".begin(), \n";
      O.indent(IndentLevel + Indent1) << " E = " << D.GenVariableName()
                                      << ".end(); B != E; ++B)\n";
      O.indent(IndentLevel) << "{\n";
      O.indent(IndentLevel + Indent1)
        << "unsigned pos = " << D.GenVariableName()
        << ".getPosition(B - " << D.GenVariableName()
        << ".begin());\n";
      O.indent(IndentLevel + Indent1)
        << "vec.push_back(std::make_pair(pos, *B));\n";
      O.indent(IndentLevel) << "}\n";
    }
  }

  void onForwardTransformedValue (const DagInit& Dag,
                                  unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 2);
    const std::string& Name = InitPtrToString(Dag.getArg(0));
    const std::string& Hook = InitPtrToString(Dag.getArg(1));
    const OptionDescription& D = OptDescs.FindParameterListOrParameter(Name);

    O.indent(IndentLevel) << "vec.push_back(std::make_pair("
                          << D.GenVariableName() << ".getPosition("
                          << (D.isList() ? "0" : "") << "), "
                          << "hooks::" << Hook << "(" << D.GenVariableName()
                          << (D.isParameter() ? ".c_str()" : "") << ")));\n";
  }

  void onNoOutFile (const DagInit& Dag,
                    unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 0);
    O.indent(IndentLevel) << "no_out_file = true;\n";
  }

  void onOutputSuffix (const DagInit& Dag,
                       unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(Dag, 1);
    this->EmitHookInvocation(InitPtrToString(Dag.getArg(0)),
                             "output_suffix = ", ";\n", IndentLevel, O);
  }

  void onStopCompilation (const DagInit& Dag,
                          unsigned IndentLevel, raw_ostream& O) const
  {
    O.indent(IndentLevel) << "stop_compilation = true;\n";
  }


  void onUnpackValues (const DagInit& Dag,
                       unsigned IndentLevel, raw_ostream& O) const
  {
    throw "'unpack_values' is deprecated. "
      "Use 'comma_separated' + 'forward_value' instead!";
  }

 public:

  explicit EmitActionHandlersCallback(const OptionDescriptions& OD)
    : OptDescs(OD)
  {
    if (!staticMembersInitialized_) {
      AddHandler("error", &EmitActionHandlersCallback::onErrorDag);
      AddHandler("warning", &EmitActionHandlersCallback::onWarningDag);
      AddHandler("append_cmd", &EmitActionHandlersCallback::onAppendCmd);
      AddHandler("forward", &EmitActionHandlersCallback::onForward);
      AddHandler("forward_as", &EmitActionHandlersCallback::onForwardAs);
      AddHandler("forward_value", &EmitActionHandlersCallback::onForwardValue);
      AddHandler("forward_transformed_value",
                 &EmitActionHandlersCallback::onForwardTransformedValue);
      AddHandler("no_out_file",
                 &EmitActionHandlersCallback::onNoOutFile);
      AddHandler("output_suffix", &EmitActionHandlersCallback::onOutputSuffix);
      AddHandler("stop_compilation",
                 &EmitActionHandlersCallback::onStopCompilation);
      AddHandler("unpack_values",
                 &EmitActionHandlersCallback::onUnpackValues);


      staticMembersInitialized_ = true;
    }
  }

  void operator()(const Init* I,
                  unsigned IndentLevel, raw_ostream& O) const
  {
    InvokeDagInitHandler(this, I, IndentLevel, O);
  }
};

void EmitGenerateActionMethodHeader(const ToolDescription& D,
                                    bool IsJoin, bool Naked,
                                    raw_ostream& O)
{
  O.indent(Indent1) << "int GenerateAction(Action& Out,\n";

  if (IsJoin)
    O.indent(Indent2) << "const PathVector& inFiles,\n";
  else
    O.indent(Indent2) << "const sys::Path& inFile,\n";

  O.indent(Indent2) << "const bool HasChildren,\n";
  O.indent(Indent2) << "const llvm::sys::Path& TempDir,\n";
  O.indent(Indent2) << "const InputLanguagesSet& InLangs,\n";
  O.indent(Indent2) << "const LanguageMap& LangMap) const\n";
  O.indent(Indent1) << "{\n";

  if (!Naked) {
    O.indent(Indent2) << "std::string cmd;\n";
    O.indent(Indent2) << "std::string out_file;\n";
    O.indent(Indent2)
      << "std::vector<std::pair<unsigned, std::string> > vec;\n";
    O.indent(Indent2) << "bool stop_compilation = !HasChildren;\n";
    O.indent(Indent2) << "bool no_out_file = false;\n";
    O.indent(Indent2) << "std::string output_suffix(\""
                      << D.OutputSuffix << "\");\n";
  }
}

// EmitGenerateActionMethod - Emit either a normal or a "join" version of the
// Tool::GenerateAction() method.
void EmitGenerateActionMethod (const ToolDescription& D,
                               const OptionDescriptions& OptDescs,
                               bool IsJoin, raw_ostream& O) {

  EmitGenerateActionMethodHeader(D, IsJoin, /* Naked = */ false, O);

  if (!D.CmdLine)
    throw "Tool " + D.Name + " has no cmd_line property!";

  // Process the 'command' property.
  O << '\n';
  EmitCmdLineVecFill(D.CmdLine, D.Name, IsJoin, Indent2, O);
  O << '\n';

  // Process the 'actions' list of this tool.
  if (D.Actions)
    EmitCaseConstructHandler(D.Actions, Indent2,
                             EmitActionHandlersCallback(OptDescs),
                             false, OptDescs, O);
  O << '\n';

  // Input file (s)
  if (!D.InFileOption.empty()) {
    O.indent(Indent2)
      << "vec.push_back(std::make_pair(InputFilenames.getPosition(0), \""
      << D.InFileOption << "\");\n";
  }

  if (IsJoin) {
    O.indent(Indent2)
      << "for (PathVector::const_iterator B = inFiles.begin(),\n";
    O.indent(Indent3) << "E = inFiles.end(); B != E; ++B)\n";
    O.indent(Indent2) << "{\n";
    O.indent(Indent3) << "vec.push_back(std::make_pair("
                      << "InputFilenames.getPosition(B - inFiles.begin()), "
                      << "B->str()));\n";
    O.indent(Indent2) << "}\n";
  }
  else {
    O.indent(Indent2) << "vec.push_back(std::make_pair("
                      << "InputFilenames.getPosition(0), inFile.str()));\n";
  }

  // Output file
  O.indent(Indent2) << "if (!no_out_file) {\n";
  if (!D.OutFileOption.empty())
    O.indent(Indent3) << "vec.push_back(std::make_pair(65536, \""
                      << D.OutFileOption << "\"));\n";

  O.indent(Indent3) << "out_file = this->OutFilename("
                    << (IsJoin ? "sys::Path(),\n" : "inFile,\n");
  O.indent(Indent4) <<
    "TempDir, stop_compilation, output_suffix.c_str()).str();\n\n";
  O.indent(Indent3) << "vec.push_back(std::make_pair(65536, out_file));\n";

  O.indent(Indent2) << "}\n\n";

  // Handle the Sink property.
  std::string SinkOption("autogenerated::");
  SinkOption += SinkOptionName;
  if (D.isSink()) {
    O.indent(Indent2) << "if (!" << SinkOption << ".empty()) {\n";
    O.indent(Indent3) << "for (cl::list<std::string>::iterator B = "
                      << SinkOption << ".begin(), E = " << SinkOption
                      << ".end(); B != E; ++B)\n";
    O.indent(Indent4) << "vec.push_back(std::make_pair(" << SinkOption
                      << ".getPosition(B - " << SinkOption
                      <<  ".begin()), *B));\n";
    O.indent(Indent2) << "}\n";
  }

  O.indent(Indent2) << "Out.Construct(cmd, this->SortArgs(vec), "
                    << "stop_compilation, out_file);\n";
  O.indent(Indent2) << "return 0;\n";
  O.indent(Indent1) << "}\n\n";
}

/// EmitGenerateActionMethods - Emit two GenerateAction() methods for
/// a given Tool class.
void EmitGenerateActionMethods (const ToolDescription& ToolDesc,
                                const OptionDescriptions& OptDescs,
                                raw_ostream& O) {
  if (!ToolDesc.isJoin()) {
    EmitGenerateActionMethodHeader(ToolDesc, /* IsJoin = */ true,
                                   /* Naked = */ true, O);
    O.indent(Indent2) << "PrintError(\"" << ToolDesc.Name
                      << " is not a Join tool!\");\n";
    O.indent(Indent2) << "return -1;\n";
    O.indent(Indent1) << "}\n\n";
  }
  else {
    EmitGenerateActionMethod(ToolDesc, OptDescs, true, O);
  }

  EmitGenerateActionMethod(ToolDesc, OptDescs, false, O);
}

/// EmitInOutLanguageMethods - Emit the [Input,Output]Language()
/// methods for a given Tool class.
void EmitInOutLanguageMethods (const ToolDescription& D, raw_ostream& O) {
  O.indent(Indent1) << "const char** InputLanguages() const {\n";
  O.indent(Indent2) << "return InputLanguages_;\n";
  O.indent(Indent1) << "}\n\n";

  O.indent(Indent1) << "const char** OutputLanguages() const {\n";
  O.indent(Indent2) << "return OutputLanguages_;\n";
  O.indent(Indent1) << "}\n\n";
}

/// EmitNameMethod - Emit the Name() method for a given Tool class.
void EmitNameMethod (const ToolDescription& D, raw_ostream& O) {
  O.indent(Indent1) << "const char* Name() const {\n";
  O.indent(Indent2) << "return \"" << D.Name << "\";\n";
  O.indent(Indent1) << "}\n\n";
}

/// EmitIsJoinMethod - Emit the IsJoin() method for a given Tool
/// class.
void EmitIsJoinMethod (const ToolDescription& D, raw_ostream& O) {
  O.indent(Indent1) << "bool IsJoin() const {\n";
  if (D.isJoin())
    O.indent(Indent2) << "return true;\n";
  else
    O.indent(Indent2) << "return false;\n";
  O.indent(Indent1) << "}\n\n";
}

/// EmitWorksOnEmptyCallback - Callback used by EmitWorksOnEmptyMethod in
/// conjunction with EmitCaseConstructHandler.
void EmitWorksOnEmptyCallback (const Init* Value,
                               unsigned IndentLevel, raw_ostream& O) {
  CheckBooleanConstant(Value);
  O.indent(IndentLevel) << "return " << Value->getAsString() << ";\n";
}

/// EmitWorksOnEmptyMethod - Emit the WorksOnEmpty() method for a given Tool
/// class.
void EmitWorksOnEmptyMethod (const ToolDescription& D,
                             const OptionDescriptions& OptDescs,
                             raw_ostream& O)
{
  O.indent(Indent1) << "bool WorksOnEmpty() const {\n";
  if (D.OnEmpty == 0)
    O.indent(Indent2) << "return false;\n";
  else
    EmitCaseConstructHandler(D.OnEmpty, Indent2, EmitWorksOnEmptyCallback,
                             /*EmitElseIf = */ true, OptDescs, O);
  O.indent(Indent1) << "}\n\n";
}

/// EmitStrArray - Emit definition of a 'const char**' static member
/// variable. Helper used by EmitStaticMemberDefinitions();
void EmitStrArray(const std::string& Name, const std::string& VarName,
                  const StrVector& StrVec, raw_ostream& O) {
  O << "const char* " << Name << "::" << VarName << "[] = {";
  for (StrVector::const_iterator B = StrVec.begin(), E = StrVec.end();
       B != E; ++B)
    O << '\"' << *B << "\", ";
  O << "0};\n";
}

/// EmitStaticMemberDefinitions - Emit static member definitions for a
/// given Tool class.
void EmitStaticMemberDefinitions(const ToolDescription& D, raw_ostream& O) {
  if (D.InLanguage.empty())
    throw "Tool " + D.Name + " has no 'in_language' property!";
  if (D.OutLanguage.empty())
    throw "Tool " + D.Name + " has no 'out_language' property!";

  EmitStrArray(D.Name, "InputLanguages_", D.InLanguage, O);
  EmitStrArray(D.Name, "OutputLanguages_", D.OutLanguage, O);
  O << '\n';
}

/// EmitToolClassDefinition - Emit a Tool class definition.
void EmitToolClassDefinition (const ToolDescription& D,
                              const OptionDescriptions& OptDescs,
                              raw_ostream& O) {
  if (D.Name == "root")
    return;

  // Header
  O << "class " << D.Name << " : public ";
  if (D.isJoin())
    O << "JoinTool";
  else
    O << "Tool";

  O << " {\nprivate:\n";
  O.indent(Indent1) << "static const char* InputLanguages_[];\n";
  O.indent(Indent1) << "static const char* OutputLanguages_[];\n\n";

  O << "public:\n";
  EmitNameMethod(D, O);
  EmitInOutLanguageMethods(D, O);
  EmitIsJoinMethod(D, O);
  EmitWorksOnEmptyMethod(D, OptDescs, O);
  EmitGenerateActionMethods(D, OptDescs, O);

  // Close class definition
  O << "};\n";

  EmitStaticMemberDefinitions(D, O);

}

/// EmitOptionDefinitions - Iterate over a list of option descriptions
/// and emit registration code.
void EmitOptionDefinitions (const OptionDescriptions& descs,
                            bool HasSink, raw_ostream& O)
{
  std::vector<OptionDescription> Aliases;

  // Emit static cl::Option variables.
  for (OptionDescriptions::const_iterator B = descs.begin(),
         E = descs.end(); B!=E; ++B) {
    const OptionDescription& val = B->second;

    if (val.Type == OptionType::Alias) {
      Aliases.push_back(val);
      continue;
    }

    O << val.GenTypeDeclaration() << ' '
      << val.GenPlainVariableName();

    O << "(\"" << val.Name << "\"\n";

    if (val.Type == OptionType::Prefix || val.Type == OptionType::PrefixList)
      O << ", cl::Prefix";

    if (val.isRequired()) {
      if (val.isList() && !val.isMultiVal())
        O << ", cl::OneOrMore";
      else
        O << ", cl::Required";
    }

    if (val.isOptional())
        O << ", cl::Optional";

    if (val.isOneOrMore())
        O << ", cl::OneOrMore";

    if (val.isZeroOrMore())
        O << ", cl::ZeroOrMore";

    if (val.isReallyHidden())
      O << ", cl::ReallyHidden";
    else if (val.isHidden())
      O << ", cl::Hidden";

    if (val.isCommaSeparated())
      O << ", cl::CommaSeparated";

    if (val.MultiVal > 1)
      O << ", cl::multi_val(" << val.MultiVal << ')';

    if (val.InitVal) {
      const std::string& str = val.InitVal->getAsString();
      O << ", cl::init(" << str << ')';
    }

    if (!val.Help.empty())
      O << ", cl::desc(\"" << val.Help << "\")";

    O << ");\n\n";
  }

  // Emit the aliases (they should go after all the 'proper' options).
  for (std::vector<OptionDescription>::const_iterator
         B = Aliases.begin(), E = Aliases.end(); B != E; ++B) {
    const OptionDescription& val = *B;

    O << val.GenTypeDeclaration() << ' '
      << val.GenPlainVariableName()
      << "(\"" << val.Name << '\"';

    const OptionDescription& D = descs.FindOption(val.Help);
    O << ", cl::aliasopt(" << D.GenVariableName() << ")";

    O << ", cl::desc(\"" << "An alias for -" + val.Help  << "\"));\n";
  }

  // Emit the sink option.
  if (HasSink)
    O << "cl::list<std::string> " << SinkOptionName << "(cl::Sink);\n";

  O << '\n';
}

/// EmitPreprocessOptionsCallback - Helper function passed to
/// EmitCaseConstructHandler() by EmitPreprocessOptions().

class EmitPreprocessOptionsCallback;

typedef void
(EmitPreprocessOptionsCallback::* EmitPreprocessOptionsCallbackHandler)
(const DagInit&, unsigned, raw_ostream&) const;

class EmitPreprocessOptionsCallback :
  public ActionHandlingCallbackBase,
  public HandlerTable<EmitPreprocessOptionsCallbackHandler>
{
  typedef EmitPreprocessOptionsCallbackHandler Handler;
  typedef void
  (EmitPreprocessOptionsCallback::* HandlerImpl)
  (const Init*, unsigned, raw_ostream&) const;

  const OptionDescriptions& OptDescs_;

  void onEachArgument(const DagInit& d, HandlerImpl h,
                      unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(d, 1);

    for (unsigned i = 0, NumArgs = d.getNumArgs(); i < NumArgs; ++i) {
      ((this)->*(h))(d.getArg(i), IndentLevel, O);
    }
  }

  void onUnsetOptionImpl(const Init* I,
                         unsigned IndentLevel, raw_ostream& O) const
  {
    const std::string& OptName = InitPtrToString(I);
    const OptionDescription& OptDesc = OptDescs_.FindOption(OptName);

    if (OptDesc.isSwitch()) {
      O.indent(IndentLevel) << OptDesc.GenVariableName() << " = false;\n";
    }
    else if (OptDesc.isParameter()) {
      O.indent(IndentLevel) << OptDesc.GenVariableName() << " = \"\";\n";
    }
    else if (OptDesc.isList()) {
      O.indent(IndentLevel) << OptDesc.GenVariableName() << ".clear();\n";
    }
    else {
      throw "Can't apply 'unset_option' to alias option '" + OptName + "'!";
    }
  }

  void onUnsetOption(const DagInit& d,
                     unsigned IndentLevel, raw_ostream& O) const
  {
    this->onEachArgument(d, &EmitPreprocessOptionsCallback::onUnsetOptionImpl,
                         IndentLevel, O);
  }

  void onSetOptionImpl(const DagInit& D,
                       unsigned IndentLevel, raw_ostream& O) const {
    CheckNumberOfArguments(D, 2);

    const std::string& OptName = InitPtrToString(D.getArg(0));
    const OptionDescription& OptDesc = OptDescs_.FindOption(OptName);
    const Init* Value = D.getArg(1);

    if (OptDesc.isList()) {
      const ListInit& List = InitPtrToList(Value);

      O.indent(IndentLevel) << OptDesc.GenVariableName() << ".clear();\n";
      for (ListInit::const_iterator B = List.begin(), E = List.end();
           B != E; ++B) {
        const Init* CurElem = *B;
        if (OptDesc.isSwitchList())
          CheckBooleanConstant(CurElem);

        O.indent(IndentLevel)
          << OptDesc.GenVariableName() << ".push_back(\""
          << (OptDesc.isSwitchList() ? CurElem->getAsString()
              : InitPtrToString(CurElem))
          << "\");\n";
      }
    }
    else if (OptDesc.isSwitch()) {
      CheckBooleanConstant(Value);
      O.indent(IndentLevel) << OptDesc.GenVariableName()
                            << " = " << Value->getAsString() << ";\n";
    }
    else if (OptDesc.isParameter()) {
      const std::string& Str = InitPtrToString(Value);
      O.indent(IndentLevel) << OptDesc.GenVariableName()
                            << " = \"" << Str << "\";\n";
    }
    else {
      throw "Can't apply 'set_option' to alias option '" + OptName + "'!";
    }
  }

  void onSetSwitch(const Init* I,
                   unsigned IndentLevel, raw_ostream& O) const {
    const std::string& OptName = InitPtrToString(I);
    const OptionDescription& OptDesc = OptDescs_.FindOption(OptName);

    if (OptDesc.isSwitch())
      O.indent(IndentLevel) << OptDesc.GenVariableName() << " = true;\n";
    else
      throw "set_option: -" + OptName + " is not a switch option!";
  }

  void onSetOption(const DagInit& d,
                   unsigned IndentLevel, raw_ostream& O) const
  {
    CheckNumberOfArguments(d, 1);

    // 2-argument form: (set_option "A", true), (set_option "B", "C"),
    // (set_option "D", ["E", "F"])
    if (d.getNumArgs() == 2) {
      const OptionDescription& OptDesc =
        OptDescs_.FindOption(InitPtrToString(d.getArg(0)));
      const Init* Opt2 = d.getArg(1);

      if (!OptDesc.isSwitch() || typeid(*Opt2) != typeid(StringInit)) {
        this->onSetOptionImpl(d, IndentLevel, O);
        return;
      }
    }

    // Multiple argument form: (set_option "A"), (set_option "B", "C", "D")
    this->onEachArgument(d, &EmitPreprocessOptionsCallback::onSetSwitch,
                         IndentLevel, O);
  }

public:

  EmitPreprocessOptionsCallback(const OptionDescriptions& OptDescs)
  : OptDescs_(OptDescs)
  {
    if (!staticMembersInitialized_) {
      AddHandler("error", &EmitPreprocessOptionsCallback::onErrorDag);
      AddHandler("warning", &EmitPreprocessOptionsCallback::onWarningDag);
      AddHandler("unset_option", &EmitPreprocessOptionsCallback::onUnsetOption);
      AddHandler("set_option", &EmitPreprocessOptionsCallback::onSetOption);

      staticMembersInitialized_ = true;
    }
  }

  void operator()(const Init* I,
                  unsigned IndentLevel, raw_ostream& O) const
  {
    InvokeDagInitHandler(this, I, IndentLevel, O);
  }

};

/// EmitPreprocessOptions - Emit the PreprocessOptions() function.
void EmitPreprocessOptions (const RecordKeeper& Records,
                            const OptionDescriptions& OptDecs, raw_ostream& O)
{
  O << "int PreprocessOptions () {\n";

  const RecordVector& OptionPreprocessors =
    Records.getAllDerivedDefinitions("OptionPreprocessor");

  for (RecordVector::const_iterator B = OptionPreprocessors.begin(),
         E = OptionPreprocessors.end(); B!=E; ++B) {
    DagInit* Case = (*B)->getValueAsDag("preprocessor");
    EmitCaseConstructHandler(Case, Indent1,
                             EmitPreprocessOptionsCallback(OptDecs),
                             false, OptDecs, O);
  }

  O << '\n';
  O.indent(Indent1) << "return 0;\n";
  O << "}\n\n";
}

class DoEmitPopulateLanguageMap;
typedef void (DoEmitPopulateLanguageMap::* DoEmitPopulateLanguageMapHandler)
(const DagInit& D);

class DoEmitPopulateLanguageMap
: public HandlerTable<DoEmitPopulateLanguageMapHandler>
{
private:
  raw_ostream& O_;

public:

  explicit DoEmitPopulateLanguageMap (raw_ostream& O) : O_(O) {
    if (!staticMembersInitialized_) {
      AddHandler("lang_to_suffixes",
                 &DoEmitPopulateLanguageMap::onLangToSuffixes);

      staticMembersInitialized_ = true;
    }
  }

  void operator() (Init* I) {
    InvokeDagInitHandler(this, I);
  }

private:

  void onLangToSuffixes (const DagInit& d) {
    CheckNumberOfArguments(d, 2);

    const std::string& Lang = InitPtrToString(d.getArg(0));
    Init* Suffixes = d.getArg(1);

    // Second argument to lang_to_suffixes is either a single string...
    if (typeid(*Suffixes) == typeid(StringInit)) {
      O_.indent(Indent1) << "langMap[\"" << InitPtrToString(Suffixes)
                         << "\"] = \"" << Lang << "\";\n";
    }
    // ...or a list of strings.
    else {
      const ListInit& Lst = InitPtrToList(Suffixes);
      assert(Lst.size() != 0);
      for (ListInit::const_iterator B = Lst.begin(), E = Lst.end();
           B != E; ++B) {
        O_.indent(Indent1) << "langMap[\"" << InitPtrToString(*B)
                           << "\"] = \"" << Lang << "\";\n";
      }
    }
  }

};

/// EmitPopulateLanguageMap - Emit the PopulateLanguageMap() function.
void EmitPopulateLanguageMap (const RecordKeeper& Records, raw_ostream& O)
{
  O << "int PopulateLanguageMap (LanguageMap& langMap) {\n";

  // For each LanguageMap:
  const RecordVector& LangMaps =
    Records.getAllDerivedDefinitions("LanguageMap");

  // Call DoEmitPopulateLanguageMap.
  for (RecordVector::const_iterator B = LangMaps.begin(),
         E = LangMaps.end(); B!=E; ++B) {
    ListInit* LangMap = (*B)->getValueAsListInit("map");
    std::for_each(LangMap->begin(), LangMap->end(),
                  DoEmitPopulateLanguageMap(O));
  }

  O << '\n';
  O.indent(Indent1) << "return 0;\n";
  O << "}\n\n";
}

/// EmitEdgePropertyHandlerCallback - Emits code that handles edge
/// properties. Helper function passed to EmitCaseConstructHandler() by
/// EmitEdgeClass().
void EmitEdgePropertyHandlerCallback (const Init* i, unsigned IndentLevel,
                                      raw_ostream& O) {
  const DagInit& d = InitPtrToDag(i);
  const std::string& OpName = GetOperatorName(d);

  if (OpName == "inc_weight") {
    O.indent(IndentLevel) << "ret += ";
  }
  else if (OpName == "error") {
    CheckNumberOfArguments(d, 1);
    O.indent(IndentLevel) << "PrintError(\""
                          << InitPtrToString(d.getArg(0))
                          << "\");\n";
    O.indent(IndentLevel) << "return -1;\n";
    return;
  }
  else {
    throw "Unknown operator in edge properties list: '" + OpName + "'!"
      "\nOnly 'inc_weight', 'dec_weight' and 'error' are allowed.";
  }

  if (d.getNumArgs() > 0)
    O << InitPtrToInt(d.getArg(0)) << ";\n";
  else
    O << "2;\n";

}

/// EmitEdgeClass - Emit a single Edge# class.
void EmitEdgeClass (unsigned N, const std::string& Target,
                    const DagInit& Case, const OptionDescriptions& OptDescs,
                    raw_ostream& O) {

  // Class constructor.
  O << "class Edge" << N << ": public Edge {\n"
    << "public:\n";
  O.indent(Indent1) << "Edge" << N << "() : Edge(\"" << Target
                    << "\") {}\n\n";

  // Function Weight().
  O.indent(Indent1)
    << "int Weight(const InputLanguagesSet& InLangs) const {\n";
  O.indent(Indent2) << "unsigned ret = 0;\n";

  // Handle the 'case' construct.
  EmitCaseConstructHandler(&Case, Indent2, EmitEdgePropertyHandlerCallback,
                           false, OptDescs, O);

  O.indent(Indent2) << "return ret;\n";
  O.indent(Indent1) << "}\n\n};\n\n";
}

/// EmitEdgeClasses - Emit Edge* classes that represent graph edges.
void EmitEdgeClasses (const DagVector& EdgeVector,
                      const OptionDescriptions& OptDescs,
                      raw_ostream& O) {
  int i = 0;
  for (DagVector::const_iterator B = EdgeVector.begin(),
         E = EdgeVector.end(); B != E; ++B) {
    const DagInit& Edge = **B;
    const std::string& Name = GetOperatorName(Edge);

    if (Name == "optional_edge") {
      assert(IsOptionalEdge(Edge));
      const std::string& NodeB = InitPtrToString(Edge.getArg(1));

      const DagInit& Weight = InitPtrToDag(Edge.getArg(2));
      EmitEdgeClass(i, NodeB, Weight, OptDescs, O);
    }
    else if (Name != "edge") {
      throw "Unknown edge class: '" + Name + "'!";
    }

    ++i;
  }
}

/// EmitPopulateCompilationGraph - Emit the PopulateCompilationGraph() function.
void EmitPopulateCompilationGraph (const DagVector& EdgeVector,
                                   const ToolDescriptions& ToolDescs,
                                   raw_ostream& O)
{
  O << "int PopulateCompilationGraph (CompilationGraph& G) {\n";

  for (ToolDescriptions::const_iterator B = ToolDescs.begin(),
         E = ToolDescs.end(); B != E; ++B)
    O.indent(Indent1) << "G.insertNode(new " << (*B)->Name << "());\n";

  O << '\n';

  // Insert edges.

  int i = 0;
  for (DagVector::const_iterator B = EdgeVector.begin(),
         E = EdgeVector.end(); B != E; ++B) {
    const DagInit& Edge = **B;
    const std::string& NodeA = InitPtrToString(Edge.getArg(0));
    const std::string& NodeB = InitPtrToString(Edge.getArg(1));

    O.indent(Indent1) << "if (int ret = G.insertEdge(\"" << NodeA << "\", ";

    if (IsOptionalEdge(Edge))
      O << "new Edge" << i << "()";
    else
      O << "new SimpleEdge(\"" << NodeB << "\")";

    O << "))\n";
    O.indent(Indent2) << "return ret;\n";

    ++i;
  }

  O << '\n';
  O.indent(Indent1) << "return 0;\n";
  O << "}\n\n";
}

/// HookInfo - Information about the hook type and number of arguments.
struct HookInfo {

  // A hook can either have a single parameter of type std::vector<std::string>,
  // or NumArgs parameters of type const char*.
  enum HookType { ListHook, ArgHook };

  HookType Type;
  unsigned NumArgs;

  HookInfo() : Type(ArgHook), NumArgs(1)
  {}

  HookInfo(HookType T) : Type(T), NumArgs(1)
  {}

  HookInfo(unsigned N) : Type(ArgHook), NumArgs(N)
  {}
};

typedef llvm::StringMap<HookInfo> HookInfoMap;

/// ExtractHookNames - Extract the hook names from all instances of
/// $CALL(HookName) in the provided command line string/action. Helper
/// function used by FillInHookNames().
class ExtractHookNames {
  HookInfoMap& HookNames_;
  const OptionDescriptions& OptDescs_;
public:
  ExtractHookNames(HookInfoMap& HookNames, const OptionDescriptions& OptDescs)
    : HookNames_(HookNames), OptDescs_(OptDescs)
  {}

  void onAction (const DagInit& Dag) {
    const std::string& Name = GetOperatorName(Dag);

    if (Name == "forward_transformed_value") {
      CheckNumberOfArguments(Dag, 2);
      const std::string& OptName = InitPtrToString(Dag.getArg(0));
      const std::string& HookName = InitPtrToString(Dag.getArg(1));
      const OptionDescription& D =
        OptDescs_.FindParameterListOrParameter(OptName);

      HookNames_[HookName] = HookInfo(D.isList() ? HookInfo::ListHook
                                      : HookInfo::ArgHook);
    }
    else if (Name == "append_cmd" || Name == "output_suffix") {
      CheckNumberOfArguments(Dag, 1);
      this->onCmdLine(InitPtrToString(Dag.getArg(0)));
    }
  }

  void onCmdLine(const std::string& Cmd) {
    StrVector cmds;
    TokenizeCmdLine(Cmd, cmds);

    for (StrVector::const_iterator B = cmds.begin(), E = cmds.end();
         B != E; ++B) {
      const std::string& cmd = *B;

      if (cmd == "$CALL") {
        unsigned NumArgs = 0;
        CheckedIncrement(B, E, "Syntax error in $CALL invocation!");
        const std::string& HookName = *B;

        if (HookName.at(0) == ')')
          throw "$CALL invoked with no arguments!";

        while (++B != E && B->at(0) != ')') {
          ++NumArgs;
        }

        HookInfoMap::const_iterator H = HookNames_.find(HookName);

        if (H != HookNames_.end() && H->second.NumArgs != NumArgs &&
            H->second.Type != HookInfo::ArgHook)
          throw "Overloading of hooks is not allowed. Overloaded hook: "
            + HookName;
        else
          HookNames_[HookName] = HookInfo(NumArgs);
      }
    }
  }

  void operator()(const Init* Arg) {

    // We're invoked on an action (either a dag or a dag list).
    if (typeid(*Arg) == typeid(DagInit)) {
      const DagInit& Dag = InitPtrToDag(Arg);
      this->onAction(Dag);
      return;
    }
    else if (typeid(*Arg) == typeid(ListInit)) {
      const ListInit& List = InitPtrToList(Arg);
      for (ListInit::const_iterator B = List.begin(), E = List.end(); B != E;
           ++B) {
        const DagInit& Dag = InitPtrToDag(*B);
        this->onAction(Dag);
      }
      return;
    }

    // We're invoked on a command line string.
    this->onCmdLine(InitPtrToString(Arg));
  }

  void operator()(const Init* Statement, unsigned) {
    this->operator()(Statement);
  }
};

/// FillInHookNames - Actually extract the hook names from all command
/// line strings. Helper function used by EmitHookDeclarations().
void FillInHookNames(const ToolDescriptions& ToolDescs,
                     const OptionDescriptions& OptDescs,
                     HookInfoMap& HookNames)
{
  // For all tool descriptions:
  for (ToolDescriptions::const_iterator B = ToolDescs.begin(),
         E = ToolDescs.end(); B != E; ++B) {
    const ToolDescription& D = *(*B);

    // Look for 'forward_transformed_value' in 'actions'.
    if (D.Actions)
      WalkCase(D.Actions, Id(), ExtractHookNames(HookNames, OptDescs));

    // Look for hook invocations in 'cmd_line'.
    if (!D.CmdLine)
      continue;
    if (dynamic_cast<StringInit*>(D.CmdLine))
      // This is a string.
      ExtractHookNames(HookNames, OptDescs).operator()(D.CmdLine);
    else
      // This is a 'case' construct.
      WalkCase(D.CmdLine, Id(), ExtractHookNames(HookNames, OptDescs));
  }
}

/// EmitHookDeclarations - Parse CmdLine fields of all the tool
/// property records and emit hook function declaration for each
/// instance of $CALL(HookName).
void EmitHookDeclarations(const ToolDescriptions& ToolDescs,
                          const OptionDescriptions& OptDescs, raw_ostream& O) {
  HookInfoMap HookNames;

  FillInHookNames(ToolDescs, OptDescs, HookNames);
  if (HookNames.empty())
    return;

  for (HookInfoMap::const_iterator B = HookNames.begin(),
         E = HookNames.end(); B != E; ++B) {
    const char* HookName = B->first();
    const HookInfo& Info = B->second;

    O.indent(Indent1) << "std::string " << HookName << "(";

    if (Info.Type == HookInfo::ArgHook) {
      for (unsigned i = 0, j = Info.NumArgs; i < j; ++i) {
        O << "const char* Arg" << i << (i+1 == j ? "" : ", ");
      }
    }
    else {
      O << "const std::vector<std::string>& Arg";
    }

    O <<");\n";
  }
}

/// EmitIncludes - Emit necessary #include directives and some
/// additional declarations.
void EmitIncludes(raw_ostream& O) {
  O << "#include \"llvm/CompilerDriver/BuiltinOptions.h\"\n"
    << "#include \"llvm/CompilerDriver/CompilationGraph.h\"\n"
    << "#include \"llvm/CompilerDriver/Error.h\"\n"
    << "#include \"llvm/CompilerDriver/Tool.h\"\n\n"

    << "#include \"llvm/Support/CommandLine.h\"\n"
    << "#include \"llvm/Support/raw_ostream.h\"\n\n"

    << "#include <algorithm>\n"
    << "#include <cstdlib>\n"
    << "#include <iterator>\n"
    << "#include <stdexcept>\n\n"

    << "using namespace llvm;\n"
    << "using namespace llvmc;\n\n"

    << "inline const char* checkCString(const char* s)\n"
    << "{ return s == NULL ? \"\" : s; }\n\n";
}


/// DriverData - Holds all information about the driver.
struct DriverData {
  OptionDescriptions OptDescs;
  ToolDescriptions ToolDescs;
  DagVector Edges;
  bool HasSink;
};

/// HasSink - Go through the list of tool descriptions and check if
/// there are any with the 'sink' property set.
bool HasSink(const ToolDescriptions& ToolDescs) {
  for (ToolDescriptions::const_iterator B = ToolDescs.begin(),
         E = ToolDescs.end(); B != E; ++B)
    if ((*B)->isSink())
      return true;

  return false;
}

/// CollectDriverData - Collect compilation graph edges, tool properties and
/// option properties from the parse tree.
void CollectDriverData (const RecordKeeper& Records, DriverData& Data) {
  // Collect option properties.
  const RecordVector& OptionLists =
    Records.getAllDerivedDefinitions("OptionList");
  CollectOptionDescriptions(OptionLists, Data.OptDescs);

  // Collect tool properties.
  const RecordVector& Tools = Records.getAllDerivedDefinitions("Tool");
  CollectToolDescriptions(Tools, Data.ToolDescs);
  Data.HasSink = HasSink(Data.ToolDescs);

  // Collect compilation graph edges.
  const RecordVector& CompilationGraphs =
    Records.getAllDerivedDefinitions("CompilationGraph");
  FillInEdgeVector(CompilationGraphs, Data.Edges);
}

/// CheckDriverData - Perform some sanity checks on the collected data.
void CheckDriverData(DriverData& Data) {
  // Filter out all tools not mentioned in the compilation graph.
  FilterNotInGraph(Data.Edges, Data.ToolDescs);

  // Typecheck the compilation graph.
  TypecheckGraph(Data.Edges, Data.ToolDescs);

  // Check that there are no options without side effects (specified
  // only in the OptionList).
  CheckForSuperfluousOptions(Data.Edges, Data.ToolDescs, Data.OptDescs);
}

void EmitDriverCode(const DriverData& Data, 
                    raw_ostream& O, RecordKeeper &Records) {
  // Emit file header.
  EmitIncludes(O);

  // Emit global option registration code.
  O << "namespace llvmc {\n"
    << "namespace autogenerated {\n\n";
  EmitOptionDefinitions(Data.OptDescs, Data.HasSink, O);
  O << "} // End namespace autogenerated.\n"
    << "} // End namespace llvmc.\n\n";

  // Emit hook declarations.
  O << "namespace hooks {\n";
  EmitHookDeclarations(Data.ToolDescs, Data.OptDescs, O);
  O << "} // End namespace hooks.\n\n";

  O << "namespace {\n\n";
  O << "using namespace llvmc::autogenerated;\n\n";

  // Emit Tool classes.
  for (ToolDescriptions::const_iterator B = Data.ToolDescs.begin(),
         E = Data.ToolDescs.end(); B!=E; ++B)
    EmitToolClassDefinition(*(*B), Data.OptDescs, O);

  // Emit Edge# classes.
  EmitEdgeClasses(Data.Edges, Data.OptDescs, O);

  O << "} // End anonymous namespace.\n\n";

  O << "namespace llvmc {\n";
  O << "namespace autogenerated {\n\n";

  // Emit PreprocessOptions() function.
  EmitPreprocessOptions(Records, Data.OptDescs, O);

  // Emit PopulateLanguageMap() function
  // (language map maps from file extensions to language names).
  EmitPopulateLanguageMap(Records, O);

  // Emit PopulateCompilationGraph() function.
  EmitPopulateCompilationGraph(Data.Edges, Data.ToolDescs, O);

  O << "} // End namespace autogenerated.\n";
  O << "} // End namespace llvmc.\n\n";

  // EOF
}


// End of anonymous namespace
}

/// run - The back-end entry point.
void LLVMCConfigurationEmitter::run (raw_ostream &O) {
  try {
    DriverData Data;

    CollectDriverData(Records, Data);
    CheckDriverData(Data);

    this->EmitSourceFileHeader("llvmc-based driver: auto-generated code", O);
    EmitDriverCode(Data, O, Records);

  } catch (std::exception& Error) {
    throw Error.what() + std::string(" - usually this means a syntax error.");
  }
}
