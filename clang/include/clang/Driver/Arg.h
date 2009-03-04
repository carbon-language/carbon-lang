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

#include "Util.h"

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
  private:
    enum ArgClass {
      PositionalArg = 0,
      JoinedArg,
      SeparateArg,
      CommaJoinedArg,
      JoinedAndSeparateArg
    };

    /// The option this argument is an instance of.
    const Option *Opt;
    
    /// The index at which this argument appears in the containing
    /// ArgList.
    unsigned Index;

  protected:
    Arg(ArgClass Kind, const Option *Opt, unsigned Index);
    
  public:
    Arg(const Arg &);

    /// render - Append the argument onto the given array as strings.
    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    virtual unsigned getNumValues() const = 0;
    virtual const char *getValue(const ArgList &Args, unsigned N) const = 0;

    const Option &getOption() const { return *Opt; }

    unsigned getIndex() const { return Index; }
  };

  /// PositionalArg - A simple positional argument.
  class PositionalArg : public Arg {
  public:
    PositionalArg(const Option *Opt, unsigned Index);

    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    virtual unsigned getNumValues() const { return 1; }
    virtual const char *getValue(const ArgList &Args, unsigned N) const;
  };

  /// JoinedArg - A single value argument where the value is joined
  /// (suffixed) to the option.
  class JoinedArg : public Arg {
  public:
    JoinedArg(const Option *Opt, unsigned Index);

    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    virtual unsigned getNumValues() const { return 1; }
    virtual const char *getValue(const ArgList &Args, unsigned N) const;
  };

  /// SeparateArg - An argument where one or more values follow the
  /// option specifier immediately in the argument vector.
  class SeparateArg : public Arg {
    unsigned NumValues;

  public:
    SeparateArg(const Option *Opt, unsigned Index, unsigned NumValues);

    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    virtual unsigned getNumValues() const { return NumValues; }
    virtual const char *getValue(const ArgList &Args, unsigned N) const;
  };

  /// CommaJoinedArg - An argument with multiple values joined by
  /// commas and joined (suffixed) to the option specifier.
  ///
  /// The key point of this arg is that it renders its values into
  /// separate arguments, which allows it to be used as a generic
  /// mechanism for passing arguments through to tools.
  class CommaJoinedArg : public Arg {
    unsigned NumValues;

  public:
    CommaJoinedArg(const Option *Opt, unsigned Index, unsigned NumValues);

    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    virtual unsigned getNumValues() const { return NumValues; }
    virtual const char *getValue(const ArgList &Args, unsigned N) const;
  };

  /// JoinedAndSeparateArg - An argument with both joined and separate
  /// values.
  class JoinedAndSeparateArg : public Arg {
  public:
    JoinedAndSeparateArg(const Option *Opt, unsigned Index);

    virtual void render(const ArgList &Args, ArgStringList &Output) const = 0;

    virtual unsigned getNumValues() const { return 2; }
    virtual const char *getValue(const ArgList &Args, unsigned N) const;
  };
} // end namespace driver
} // end namespace clang

#endif
