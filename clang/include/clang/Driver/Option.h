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

#include "clang/Driver/OptSpecifier.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Basic/LLVM.h"

namespace clang {
namespace driver {
  class Arg;
  class ArgList;

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

    enum RenderStyleKind {
      RenderCommaJoinedStyle,
      RenderJoinedStyle,
      RenderSeparateStyle,
      RenderValuesStyle
    };

  private:
    OptionClass Kind;

    /// The option ID.
    OptSpecifier ID;

    /// The option name.
    StringRef Name;

    /// Group this option is a member of, if any.
    const Option *Group;

    /// Option that this is an alias for, if any.
    const Option *Alias;

    unsigned NumArgs;

    /// Unsupported options will be rejected.
    bool Unsupported : 1;

    /// Treat this option like a linker input?
    bool LinkerInput : 1;

    /// When rendering as an input, don't render the option.

    // FIXME: We should ditch the render/renderAsInput distinction.
    bool NoOptAsInput : 1;

    /// The style to using when rendering arguments parsed by this option.
    unsigned RenderStyle : 2;

    /// This option is only consumed by the driver.
    bool DriverOption : 1;

    /// This option should not report argument unused errors.
    bool NoArgumentUnused : 1;

    /// This option should not be implicitly forwarded.
    bool NoForward : 1;

    /// CC1Option - This option should be accepted by clang -cc1.
    bool CC1Option : 1;

  public:
    Option(OptionClass Kind, OptSpecifier ID, const char *Name,
           const Option *Group, const Option *Alias, unsigned Args);
    ~Option();

    unsigned getID() const { return ID.getID(); }
    OptionClass getKind() const { return Kind; }
    StringRef getName() const { return Name; }
    const Option *getGroup() const { return Group; }
    const Option *getAlias() const { return Alias; }

    bool isUnsupported() const { return Unsupported; }
    void setUnsupported(bool Value) { Unsupported = Value; }

    bool isLinkerInput() const { return LinkerInput; }
    void setLinkerInput(bool Value) { LinkerInput = Value; }

    bool hasNoOptAsInput() const { return NoOptAsInput; }
    void setNoOptAsInput(bool Value) { NoOptAsInput = Value; }

    RenderStyleKind getRenderStyle() const {
      return RenderStyleKind(RenderStyle);
    }
    void setRenderStyle(RenderStyleKind Value) { RenderStyle = Value; }

    bool isDriverOption() const { return DriverOption; }
    void setDriverOption(bool Value) { DriverOption = Value; }

    bool hasNoArgumentUnused() const { return NoArgumentUnused; }
    void setNoArgumentUnused(bool Value) { NoArgumentUnused = Value; }

    bool hasNoForward() const { return NoForward; }
    void setNoForward(bool Value) { NoForward = Value; }

    bool isCC1Option() const { return CC1Option; }
    void setIsCC1Option(bool Value) { CC1Option = Value; }

    bool hasForwardToGCC() const {
      return !NoForward && !DriverOption && !LinkerInput;
    }

    /// getUnaliasedOption - Return the final option this option
    /// aliases (itself, if the option has no alias).
    const Option *getUnaliasedOption() const {
      if (Alias) return Alias->getUnaliasedOption();
      return this;
    }

    /// getRenderName - Return the name to use when rendering this
    /// option.
    StringRef getRenderName() const {
      return getUnaliasedOption()->getName();
    }

    /// matches - Predicate for whether this option is part of the
    /// given option (which may be a group).
    ///
    /// Note that matches against options which are an alias should never be
    /// done -- aliases do not participate in matching and so such a query will
    /// always be false.
    bool matches(OptSpecifier ID) const;

    /// accept - Potentially accept the current argument, returning a
    /// new Arg instance, or 0 if the option does not accept this
    /// argument (or the argument is missing values).
    ///
    /// If the option accepts the current argument, accept() sets
    /// Index to the position where argument parsing should resume
    /// (even if the argument is missing values).
    Arg *accept(const ArgList &Args, unsigned &Index) const;

    void dump() const;
  };

} // end namespace driver
} // end namespace clang

#endif
