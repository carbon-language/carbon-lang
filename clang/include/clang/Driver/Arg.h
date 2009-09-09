//===--- Arg.h - Parsed Argument Classes ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_ARG_H_
#define CLANG_DRIVER_ARG_H_

#include "llvm/Support/Casting.h"
using llvm::isa;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;

#include "Util.h"
#include <vector>
#include <string>

namespace clang {
namespace driver {
  class ArgList;
  class Option;

  /// Arg - A concrete instance of a particular driver option.
  ///
  /// The Arg class encodes just enough information to be able to
  /// derive the argument values efficiently. In addition, Arg
  /// instances have an intrusive double linked list which is used by
  /// ArgList to provide efficient iteration over all instances of a
  /// particular option.
  class Arg {
  public:
    enum ArgClass {
      FlagClass = 0,
      PositionalClass,
      JoinedClass,
      SeparateClass,
      CommaJoinedClass,
      JoinedAndSeparateClass
    };

  private:
    ArgClass Kind;

    /// The option this argument is an instance of.
    const Option *Opt;

    /// The argument this argument was derived from (during tool chain
    /// argument translation), if any.
    const Arg *BaseArg;

    /// The index at which this argument appears in the containing
    /// ArgList.
    unsigned Index;

    /// Flag indicating whether this argument was used to effect
    /// compilation; used for generating "argument unused"
    /// diagnostics.
    mutable bool Claimed;

  protected:
    Arg(ArgClass Kind, const Option *Opt, unsigned Index,
        const Arg *BaseArg = 0);

  public:
    Arg(const Arg &);
    virtual ~Arg();

    ArgClass getKind() const { return Kind; }
    const Option &getOption() const { return *Opt; }
    unsigned getIndex() const { return Index; }

    /// getBaseArg - Return the base argument which generated this
    /// arg; this is either the argument itself or the argument it was
    /// derived from during tool chain specific argument translation.
    const Arg &getBaseArg() const {
      return BaseArg ? *BaseArg : *this;
    }
    void setBaseArg(const Arg *_BaseArg) {
      BaseArg = _BaseArg;
    }

    bool isClaimed() const { return getBaseArg().Claimed; }

    /// claim - Set the Arg claimed bit.

    // FIXME: We need to deal with derived arguments and set the bit
    // in the original argument; not the derived one.
    void claim() const { getBaseArg().Claimed = true; }

    virtual unsigned getNumValues() const = 0;
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const = 0;

    /// render - Append the argument onto the given array as strings.
    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    /// renderAsInput - Append the argument, render as an input, onto
    /// the given array as strings. The distinction is that some
    /// options only render their values when rendered as a input
    /// (e.g., Xlinker).
    void renderAsInput(const ArgList &Args, ArgStringList &Output) const;

    static bool classof(const Arg *) { return true; }

    void dump() const;

    /// getAsString - Return a formatted version of the argument and
    /// its values, for debugging and diagnostics.
    std::string getAsString(const ArgList &Args) const;
  };

  /// FlagArg - An argument with no value.
  class FlagArg : public Arg {
  public:
    FlagArg(const Option *Opt, unsigned Index, const Arg *BaseArg = 0);

    virtual void render(const ArgList &Args, ArgStringList &Output) const;

    virtual unsigned getNumValues() const { return 0; }
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const;

    static bool classof(const Arg *A) {
      return A->getKind() == Arg::FlagClass;
    }
    static bool classof(const FlagArg *) { return true; }
  };

  /// PositionalArg - A simple positional argument.
  class PositionalArg : public Arg {
  public:
    PositionalArg(const Option *Opt, unsigned Index, const Arg *BaseArg = 0);

    virtual void render(const ArgList &Args, ArgStringList &Output) const;

    virtual unsigned getNumValues() const { return 1; }
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const;

    static bool classof(const Arg *A) {
      return A->getKind() == Arg::PositionalClass;
    }
    static bool classof(const PositionalArg *) { return true; }
  };

  /// JoinedArg - A single value argument where the value is joined
  /// (suffixed) to the option.
  class JoinedArg : public Arg {
  public:
    JoinedArg(const Option *Opt, unsigned Index, const Arg *BaseArg = 0);

    virtual void render(const ArgList &Args, ArgStringList &Output) const;

    virtual unsigned getNumValues() const { return 1; }
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const;

    static bool classof(const Arg *A) {
      return A->getKind() == Arg::JoinedClass;
    }
    static bool classof(const JoinedArg *) { return true; }
  };

  /// SeparateArg - An argument where one or more values follow the
  /// option specifier immediately in the argument vector.
  class SeparateArg : public Arg {
    unsigned NumValues;

  public:
    SeparateArg(const Option *Opt, unsigned Index, unsigned NumValues,
                const Arg *BaseArg = 0);

    virtual void render(const ArgList &Args, ArgStringList &Output) const;

    virtual unsigned getNumValues() const { return NumValues; }
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const;

    static bool classof(const Arg *A) {
      return A->getKind() == Arg::SeparateClass;
    }
    static bool classof(const SeparateArg *) { return true; }
  };

  /// CommaJoinedArg - An argument with multiple values joined by
  /// commas and joined (suffixed) to the option specifier.
  ///
  /// The key point of this arg is that it renders its values into
  /// separate arguments, which allows it to be used as a generic
  /// mechanism for passing arguments through to tools.
  class CommaJoinedArg : public Arg {
    std::vector<std::string> Values;

  public:
    CommaJoinedArg(const Option *Opt, unsigned Index, const char *Str,
                   const Arg *BaseArg = 0);

    virtual void render(const ArgList &Args, ArgStringList &Output) const;

    virtual unsigned getNumValues() const { return Values.size(); }
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const;

    static bool classof(const Arg *A) {
      return A->getKind() == Arg::CommaJoinedClass;
    }
    static bool classof(const CommaJoinedArg *) { return true; }
  };

  /// JoinedAndSeparateArg - An argument with both joined and separate
  /// values.
  class JoinedAndSeparateArg : public Arg {
  public:
    JoinedAndSeparateArg(const Option *Opt, unsigned Index,
                         const Arg *BaseArg = 0);

    virtual void render(const ArgList &Args, ArgStringList &Output) const;

    virtual unsigned getNumValues() const { return 2; }
    virtual const char *getValue(const ArgList &Args, unsigned N=0) const;

    static bool classof(const Arg *A) {
      return A->getKind() == Arg::JoinedAndSeparateClass;
    }
    static bool classof(const JoinedAndSeparateArg *) { return true; }
  };
} // end namespace driver
} // end namespace clang

#endif
