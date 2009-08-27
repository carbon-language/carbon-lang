//===--- ArgList.h - Argument List Management ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_ARGLIST_H_
#define CLANG_DRIVER_ARGLIST_H_

#include "clang/Driver/Options.h"

#include "clang/Driver/Util.h"
#include "llvm/ADT/SmallVector.h"

#include <list>
#include <string>

namespace clang {
namespace driver {
  class Arg;

  /// ArgList - Ordered collection of driver arguments.
  ///
  /// The ArgList class manages a list of Arg instances as well as
  /// auxiliary data and convenience methods to allow Tools to quickly
  /// check for the presence of Arg instances for a particular Option
  /// and to iterate over groups of arguments.
  class ArgList {
  public:
    typedef llvm::SmallVector<Arg*, 16> arglist_type;
    typedef arglist_type::iterator iterator;
    typedef arglist_type::const_iterator const_iterator;
    typedef arglist_type::reverse_iterator reverse_iterator;
    typedef arglist_type::const_reverse_iterator const_reverse_iterator;

  private:
    /// The full list of arguments.
    arglist_type &Args;

  protected:
    ArgList(arglist_type &Args);

  public:
    virtual ~ArgList();

    /// @name Arg Access
    /// @{

    /// append - Append \arg A to the arg list.
    void append(Arg *A);

    arglist_type &getArgs() { return Args; }
    const arglist_type &getArgs() const { return Args; }

    unsigned size() const { return Args.size(); }

    iterator begin() { return Args.begin(); }
    iterator end() { return Args.end(); }

    reverse_iterator rbegin() { return Args.rbegin(); }
    reverse_iterator rend() { return Args.rend(); }

    const_iterator begin() const { return Args.begin(); }
    const_iterator end() const { return Args.end(); }
    
    const_reverse_iterator rbegin() const { return Args.rbegin(); }
    const_reverse_iterator rend() const { return Args.rend(); }

    /// hasArg - Does the arg list contain any option matching \arg Id.
    ///
    /// \arg Claim Whether the argument should be claimed, if it exists.
    bool hasArg(options::ID Id, bool Claim=true) const { 
      return getLastArg(Id, Claim) != 0;
    }
    bool hasArg(options::ID Id0, options::ID Id1, bool Claim=true) const { 
      return getLastArg(Id0, Id1, Claim) != 0;
    }

    /// getLastArg - Return the last argument matching \arg Id, or null.
    ///
    /// \arg Claim Whether the argument should be claimed, if it exists.
    Arg *getLastArg(options::ID Id, bool Claim=true) const;
    Arg *getLastArg(options::ID Id0, options::ID Id1, bool Claim=true) const;
    Arg *getLastArg(options::ID Id0, options::ID Id1, options::ID Id2,
                    bool Claim=true) const;

    /// getArgString - Return the input argument string at \arg Index.
    virtual const char *getArgString(unsigned Index) const = 0;
    /// @name Translation Utilities
    /// @{

    /// hasFlag - Given an option \arg Pos and its negative form \arg
    /// Neg, return true if the option is present, false if the
    /// negation is present, and \arg Default if neither option is
    /// given. If both the option and its negation are present, the
    /// last one wins.
    bool hasFlag(options::ID Pos, options::ID Neg, bool Default=true) const;

    /// AddLastArg - Render only the last argument match \arg Id0, if
    /// present.
    void AddLastArg(ArgStringList &Output, options::ID Id0) const;

    /// AddAllArgs - Render all arguments matching the given ids.
    void AddAllArgs(ArgStringList &Output, options::ID Id0) const;
    void AddAllArgs(ArgStringList &Output, options::ID Id0, 
                    options::ID Id1) const;
    void AddAllArgs(ArgStringList &Output, options::ID Id0, options::ID Id1, 
                    options::ID Id2) const;

    /// AddAllArgValues - Render the argument values of all arguments
    /// matching the given ids.
    void AddAllArgValues(ArgStringList &Output, options::ID Id0) const;
    void AddAllArgValues(ArgStringList &Output, options::ID Id0, 
                         options::ID Id1) const;

    /// AddAllArgsTranslated - Render all the arguments matching the
    /// given ids, but forced to separate args and using the provided
    /// name instead of the first option value.
    ///
    /// \param Joined - If true, render the argument as joined with
    /// the option specifier.
    void AddAllArgsTranslated(ArgStringList &Output, options::ID Id0,
                              const char *Translation, 
                              bool Joined = false) const;

    /// ClaimAllArgs - Claim all arguments which match the given
    /// option id.
    void ClaimAllArgs(options::ID Id0) const;

    /// @}
    /// @name Arg Synthesis
    /// @{

    /// MakeArgString - Construct a constant string pointer whose
    /// lifetime will match that of the ArgList.
    virtual const char *MakeArgString(const char *Str) const = 0;

    /// @}
  };

  class InputArgList : public ArgList  {
  private:
    /// The internal list of arguments.
    arglist_type ActualArgs;

    /// List of argument strings used by the contained Args.
    ///
    /// This is mutable since we treat the ArgList as being the list
    /// of Args, and allow routines to add new strings (to have a
    /// convenient place to store the memory) via MakeIndex.
    mutable ArgStringList ArgStrings;

    /// Strings for synthesized arguments.
    ///
    /// This is mutable since we treat the ArgList as being the list
    /// of Args, and allow routines to add new strings (to have a
    /// convenient place to store the memory) via MakeIndex.
    mutable std::list<std::string> SynthesizedStrings;

    /// The number of original input argument strings.
    unsigned NumInputArgStrings;

  public:
    InputArgList(const char **ArgBegin, const char **ArgEnd);
    InputArgList(const ArgList &);
    ~InputArgList();

    virtual const char *getArgString(unsigned Index) const { 
      return ArgStrings[Index]; 
    }

    /// getNumInputArgStrings - Return the number of original input
    /// argument strings.
    unsigned getNumInputArgStrings() const { return NumInputArgStrings; }

    /// @name Arg Synthesis
    /// @{

  public:
    /// MakeIndex - Get an index for the given string(s).
    unsigned MakeIndex(const char *String0) const;
    unsigned MakeIndex(const char *String0, const char *String1) const;

    virtual const char *MakeArgString(const char *Str) const;

    /// @}
  };

  /// DerivedArgList - An ordered collection of driver arguments,
  /// whose storage may be in another argument list.
  class DerivedArgList : public ArgList {
    InputArgList &BaseArgs;

    /// The internal list of arguments.
    arglist_type ActualArgs;

    /// The list of arguments we synthesized.
    arglist_type SynthesizedArgs;

    /// Is this only a proxy for the base ArgList?
    bool OnlyProxy;

  public:
    /// Construct a new derived arg list from \arg BaseArgs.
    ///
    /// \param OnlyProxy - If true, this is only a proxy for the base
    /// list (to adapt the type), and it's Args list is unused.
    DerivedArgList(InputArgList &BaseArgs, bool OnlyProxy);
    ~DerivedArgList();

    virtual const char *getArgString(unsigned Index) const {
      return BaseArgs.getArgString(Index); 
    }

    /// @name Arg Synthesis
    /// @{

    virtual const char *MakeArgString(const char *Str) const;

    /// MakeFlagArg - Construct a new FlagArg for the given option
    /// \arg Id.
    Arg *MakeFlagArg(const Arg *BaseArg, const Option *Opt) const;

    /// MakePositionalArg - Construct a new Positional arg for the
    /// given option \arg Id, with the provided \arg Value.
    Arg *MakePositionalArg(const Arg *BaseArg, const Option *Opt, 
                           const char *Value) const;

    /// MakeSeparateArg - Construct a new Positional arg for the
    /// given option \arg Id, with the provided \arg Value.
    Arg *MakeSeparateArg(const Arg *BaseArg, const Option *Opt, 
                         const char *Value) const;

    /// MakeJoinedArg - Construct a new Positional arg for the
    /// given option \arg Id, with the provided \arg Value.
    Arg *MakeJoinedArg(const Arg *BaseArg, const Option *Opt, 
                       const char *Value) const;

    /// @}
  };

} // end namespace driver
} // end namespace clang

#endif
