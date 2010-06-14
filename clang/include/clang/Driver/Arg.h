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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
    Arg(const Arg &); // DO NOT IMPLEMENT
    void operator=(const Arg &); // DO NOT IMPLEMENT

  private:
    /// The option this argument is an instance of.
    const Option *Opt;

    /// The argument this argument was derived from (during tool chain
    /// argument translation), if any.
    const Arg *BaseArg;

    /// The index at which this argument appears in the containing
    /// ArgList.
    unsigned Index;

    /// Was this argument used to effect compilation; used for generating
    /// "argument unused" diagnostics.
    mutable unsigned Claimed : 1;

    /// Does this argument own its values.
    mutable unsigned OwnsValues : 1;

    /// The argument values, as C strings.
    llvm::SmallVector<const char *, 2> Values;

  public:
    Arg(const Option *Opt, unsigned Index, const Arg *BaseArg = 0);
    Arg(const Option *Opt, unsigned Index,
        const char *Value0, const Arg *BaseArg = 0);
    Arg(const Option *Opt, unsigned Index,
        const char *Value0, const char *Value1, const Arg *BaseArg = 0);
    ~Arg();

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

    bool getOwnsValues() const { return OwnsValues; }
    void setOwnsValues(bool Value) const { OwnsValues = Value; }

    bool isClaimed() const { return getBaseArg().Claimed; }

    /// claim - Set the Arg claimed bit.
    void claim() const { getBaseArg().Claimed = true; }

    unsigned getNumValues() const { return Values.size(); }
    const char *getValue(const ArgList &Args, unsigned N=0) const {
      return Values[N];
    }

    llvm::SmallVectorImpl<const char*> &getValues() {
      return Values;
    }

    bool containsValue(llvm::StringRef Value) const {
      for (unsigned i = 0, e = getNumValues(); i != e; ++i)
        if (Values[i] == Value)
          return true;
      return false;
    }

    /// render - Append the argument onto the given array as strings.
    void render(const ArgList &Args, ArgStringList &Output) const;

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

} // end namespace driver
} // end namespace clang

#endif
