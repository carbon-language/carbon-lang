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

#include "clang/Driver/OptTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "clang/Basic/LLVM.h"

namespace clang {
namespace driver {
  class Arg;
  class ArgList;

namespace options {
  enum DriverFlag {
    DriverOption     = (1 << 0),
    HelpHidden       = (1 << 1),
    LinkerInput      = (1 << 2),
    NoArgumentUnused = (1 << 3),
    NoForward        = (1 << 4),
    RenderAsInput    = (1 << 5),
    RenderJoined     = (1 << 6),
    RenderSeparate   = (1 << 7),
    Unsupported      = (1 << 8),
    CC1Option        = (1 << 9)
  };
}

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
    const OptTable::Info *Info;
    const OptTable *Owner;

  public:
    Option(const OptTable::Info *Info, const OptTable *Owner);
    ~Option();

    bool isValid() const {
      return Info != 0;
    }

    unsigned getID() const {
      assert(Info && "Must have a valid info!");
      return Info->ID;
    }
    
    OptionClass getKind() const {
      assert(Info && "Must have a valid info!");
      return OptionClass(Info->Kind);
    }
    
    StringRef getName() const {
      assert(Info && "Must have a valid info!");
      return Info->Name;
    }
    
    const Option getGroup() const {
      assert(Info && "Must have a valid info!");
      assert(Owner && "Must have a valid owner!");
      return Owner->getOption(Info->GroupID);
    }
    
    const Option getAlias() const {
      assert(Info && "Must have a valid info!");
      assert(Owner && "Must have a valid owner!");
      return Owner->getOption(Info->AliasID);
    }

    unsigned getNumArgs() const { return Info->Param; }

    bool isUnsupported() const { return Info->Flags & options::Unsupported; }

    bool isLinkerInput() const { return Info->Flags & options::LinkerInput; }

    bool hasNoOptAsInput() const { return Info->Flags & options::RenderAsInput;}

    RenderStyleKind getRenderStyle() const {
      if (Info->Flags & options::RenderJoined)
        return RenderJoinedStyle;
      if (Info->Flags & options::RenderSeparate)
        return RenderSeparateStyle;
      switch (getKind()) {
      case GroupClass:
      case InputClass:
      case UnknownClass:
        return RenderValuesStyle;
      case JoinedClass:
      case JoinedAndSeparateClass:
        return RenderJoinedStyle;
      case CommaJoinedClass:
        return RenderCommaJoinedStyle;
      case FlagClass:
      case SeparateClass:
      case MultiArgClass:
      case JoinedOrSeparateClass:
        return RenderSeparateStyle;
      }
      llvm_unreachable("Unexpected kind!");
    }

    bool isDriverOption() const { return Info->Flags & options::DriverOption; }

    bool hasNoArgumentUnused() const {
      return Info->Flags & options::NoArgumentUnused;
    }

    bool hasNoForward() const { return Info->Flags & options::NoForward; }

    bool isCC1Option() const { return Info->Flags & options::CC1Option; }

    bool hasForwardToGCC() const {
      return !hasNoForward() && !isDriverOption() && !isLinkerInput();
    }

    /// getUnaliasedOption - Return the final option this option
    /// aliases (itself, if the option has no alias).
    const Option getUnaliasedOption() const {
      const Option Alias = getAlias();
      if (Alias.isValid()) return Alias.getUnaliasedOption();
      return *this;
    }

    /// getRenderName - Return the name to use when rendering this
    /// option.
    StringRef getRenderName() const {
      return getUnaliasedOption().getName();
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
