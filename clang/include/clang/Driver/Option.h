//===--- Option.h - Abstract Driver Options ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_OPTION_H_
#define CLANG_DRIVER_OPTION_H_

namespace clang {
namespace driver {
  class Arg;
  class ArgList;
  class OptionGroup;
  
  /// Option - Abstract representation for a single form of driver
  /// argument.
  ///
  /// An Option class represents a form of option that the driver
  /// takes, for example how many arguments the option has and how
  /// they can be provided. Individual option instances store
  /// additional information about what group the option is a member
  /// of (if any), if the option is an alias, and a number of
  /// flags. At runtime the driver parses the command line into
  /// concrete Arg instances, each of which corresponds to a
  /// particular Option instance.
  class Option {
  public:
    enum OptionClass {
      GroupOption = 0,
      InputOption,
      UnknownOption,
      FlagOption,
      JoinedOption,
      SeparateOption,
      CommaJoinedOption,
      MultiArgOption,
      JoinedOrSeparateOption,
      JoinedAndSeparateOption
    };

  private:
    OptionClass Kind;

    /// The option name.
    const char *Name; 

    /// Group this option is a member of, if any.
    const OptionGroup *Group; 

    /// Option that this is an alias for, if any.
    const Option *Alias;

  protected:
    Option(OptionClass Kind, const char *Name, 
           OptionGroup *Group, Option *Alias);
  public:
    virtual ~Option();

    OptionClass getKind() const { return Kind; }
    const char *getName() const { return Name; }
    const OptionGroup *getGroup() const { return Group; }
    const Option *getAlias() const { return Alias; }

    /// getUnaliasedOption - Return the final option this option
    /// aliases (itself, if the option has no alias).
    const Option *getUnaliasedOption() const { 
      if (Alias) return Alias->getUnaliasedOption();
      return this;
    }

    /// getRenderName - Return the name to use when rendering this
    /// option.
    const char *getRenderName() const {
      return getUnaliasedOption()->getName();
    }

    /// matches - Predicate for whether this option is part of the
    /// given option (which may be a group).
    bool matches(const Option *Opt) const;

    /// accept - Potentially accept the current argument, returning a
    /// new Arg instance, or 0 if the option does not accept this
    /// argument.
    ///
    /// May issue a missing argument error.
    virtual Arg *accept(ArgList &Args, unsigned Index) const = 0;
  };
  
  /// OptionGroup - A set of options which are can be handled uniformly
  /// by the driver.
  class OptionGroup : public Option {
    OptionGroup *Group;
  
  public:
    OptionGroup(const char *Name, OptionGroup *Group);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };
  
  // Dummy option classes.
  
  /// InputOption - Dummy option class for representing driver inputs.
  class InputOption : public Option {
  public:
    InputOption();

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  /// UnknownOption - Dummy option class for represent unknown arguments.
  class UnknownOption : public Option {
  public:
    UnknownOption();

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  // Normal options.

  class FlagOption : public Option {
  public:
    FlagOption(const char *Name, OptionGroup *Group, Option *Alias);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  class JoinedOption : public Option {
    JoinedOption(const char *Name, OptionGroup *Group, Option *Alias);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  class CommaJoinedOption : public Option {
    CommaJoinedOption(const char *Name, OptionGroup *Group, Option *Alias);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  class SeparateOption : public Option {
    SeparateOption(const char *Name, OptionGroup *Group, Option *Alias);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  /// MultiArgOption - An option which takes multiple arguments (these
  /// are always separate arguments).
  class MultiArgOption : public Option {
    unsigned NumArgs;

  public:
    MultiArgOption(const char *Name, OptionGroup *Group, Option *Alias,
                   unsigned NumArgs);

    unsigned getNumArgs() const { return NumArgs; }

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  /// JoinedOrSeparateOption - An option which either literally
  /// prefixes its (non-empty) value, or is follwed by a value.
  class JoinedOrSeparateOption : public Option {
    JoinedOrSeparateOption(const char *Name, OptionGroup *Group, Option *Alias);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

  /// JoinedAndSeparateOption - An option which literally prefixes its
  /// value and is followed by another value.
  class JoinedAndSeparateOption : public Option {
    JoinedAndSeparateOption(const char *Name, OptionGroup *Group, Option *Alias);

    virtual Arg *accept(ArgList &Args, unsigned Index) const;
  };

} // end namespace driver
} // end namespace clang

#endif
