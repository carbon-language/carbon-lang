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

#include "Options.h"

#include "llvm/Support/Casting.h"
using llvm::isa;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;

namespace clang {
namespace driver {
  class Arg;
  class InputArgList;
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
      GroupClass = 0,
      InputClass,
      UnknownClass,
      FlagClass,
      JoinedClass,
      SeparateClass,
      CommaJoinedClass,
      MultiArgClass,
      JoinedOrSeparateClass,
      JoinedAndSeparateClass
    };

  private:
    OptionClass Kind;

    options::ID ID;

    /// The option name.
    const char *Name;

    /// Group this option is a member of, if any.
    const OptionGroup *Group;

    /// Option that this is an alias for, if any.
    const Option *Alias;

    /// Unsupported options will not be rejected.
    bool Unsupported : 1;

    /// Treat this option like a linker input?
    bool LinkerInput : 1;

    /// When rendering as an input, don't render the option.

    // FIXME: We should ditch the render/renderAsInput distinction.
    bool NoOptAsInput : 1;

    /// Always render this option as separate form its value.
    bool ForceSeparateRender : 1;

    /// Always render this option joined with its value.
    bool ForceJoinedRender : 1;

    /// This option is only consumed by the driver.
    bool DriverOption : 1;

    /// This option should not report argument unused errors.
    bool NoArgumentUnused : 1;

  protected:
    Option(OptionClass Kind, options::ID ID, const char *Name,
           const OptionGroup *Group, const Option *Alias);
  public:
    virtual ~Option();

    options::ID getId() const { return ID; }
    OptionClass getKind() const { return Kind; }
    const char *getName() const { return Name; }
    const OptionGroup *getGroup() const { return Group; }
    const Option *getAlias() const { return Alias; }

    bool isUnsupported() const { return Unsupported; }
    void setUnsupported(bool Value) { Unsupported = Value; }

    bool isLinkerInput() const { return LinkerInput; }
    void setLinkerInput(bool Value) { LinkerInput = Value; }

    bool hasNoOptAsInput() const { return NoOptAsInput; }
    void setNoOptAsInput(bool Value) { NoOptAsInput = Value; }

    bool hasForceSeparateRender() const { return ForceSeparateRender; }
    void setForceSeparateRender(bool Value) { ForceSeparateRender = Value; }

    bool hasForceJoinedRender() const { return ForceJoinedRender; }
    void setForceJoinedRender(bool Value) { ForceJoinedRender = Value; }

    bool isDriverOption() const { return DriverOption; }
    void setDriverOption(bool Value) { DriverOption = Value; }

    bool hasNoArgumentUnused() const { return NoArgumentUnused; }
    void setNoArgumentUnused(bool Value) { NoArgumentUnused = Value; }

    bool hasForwardToGCC() const { return !DriverOption && !LinkerInput; }

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
    bool matches(options::ID Id) const;

    /// accept - Potentially accept the current argument, returning a
    /// new Arg instance, or 0 if the option does not accept this
    /// argument (or the argument is missing values).
    ///
    /// If the option accepts the current argument, accept() sets
    /// Index to the position where argument parsing should resume
    /// (even if the argument is missing values).
    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const = 0;

    void dump() const;

    static bool classof(const Option *) { return true; }
  };

  /// OptionGroup - A set of options which are can be handled uniformly
  /// by the driver.
  class OptionGroup : public Option {
  public:
    OptionGroup(options::ID ID, const char *Name, const OptionGroup *Group);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::GroupClass;
    }
    static bool classof(const OptionGroup *) { return true; }
  };

  // Dummy option classes.

  /// InputOption - Dummy option class for representing driver inputs.
  class InputOption : public Option {
  public:
    InputOption();

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::InputClass;
    }
    static bool classof(const InputOption *) { return true; }
  };

  /// UnknownOption - Dummy option class for represent unknown arguments.
  class UnknownOption : public Option {
  public:
    UnknownOption();

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::UnknownClass;
    }
    static bool classof(const UnknownOption *) { return true; }
  };

  // Normal options.

  class FlagOption : public Option {
  public:
    FlagOption(options::ID ID, const char *Name, const OptionGroup *Group,
               const Option *Alias);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::FlagClass;
    }
    static bool classof(const FlagOption *) { return true; }
  };

  class JoinedOption : public Option {
  public:
    JoinedOption(options::ID ID, const char *Name, const OptionGroup *Group,
                 const Option *Alias);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::JoinedClass;
    }
    static bool classof(const JoinedOption *) { return true; }
  };

  class SeparateOption : public Option {
  public:
    SeparateOption(options::ID ID, const char *Name, const OptionGroup *Group,
                   const Option *Alias);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::SeparateClass;
    }
    static bool classof(const SeparateOption *) { return true; }
  };

  class CommaJoinedOption : public Option {
  public:
    CommaJoinedOption(options::ID ID, const char *Name,
                      const OptionGroup *Group, const Option *Alias);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::CommaJoinedClass;
    }
    static bool classof(const CommaJoinedOption *) { return true; }
  };

  // FIXME: Fold MultiArgOption into SeparateOption?

  /// MultiArgOption - An option which takes multiple arguments (these
  /// are always separate arguments).
  class MultiArgOption : public Option {
    unsigned NumArgs;

  public:
    MultiArgOption(options::ID ID, const char *Name, const OptionGroup *Group,
                   const Option *Alias, unsigned NumArgs);

    unsigned getNumArgs() const { return NumArgs; }

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::MultiArgClass;
    }
    static bool classof(const MultiArgOption *) { return true; }
  };

  /// JoinedOrSeparateOption - An option which either literally
  /// prefixes its (non-empty) value, or is follwed by a value.
  class JoinedOrSeparateOption : public Option {
  public:
    JoinedOrSeparateOption(options::ID ID, const char *Name,
                           const OptionGroup *Group, const Option *Alias);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::JoinedOrSeparateClass;
    }
    static bool classof(const JoinedOrSeparateOption *) { return true; }
  };

  /// JoinedAndSeparateOption - An option which literally prefixes its
  /// value and is followed by another value.
  class JoinedAndSeparateOption : public Option {
  public:
    JoinedAndSeparateOption(options::ID ID, const char *Name,
                            const OptionGroup *Group, const Option *Alias);

    virtual Arg *accept(const InputArgList &Args, unsigned &Index) const;

    static bool classof(const Option *O) {
      return O->getKind() == Option::JoinedAndSeparateClass;
    }
    static bool classof(const JoinedAndSeparateOption *) { return true; }
  };

} // end namespace driver
} // end namespace clang

#endif
