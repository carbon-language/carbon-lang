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

#include "Util.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

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

  private:
    /// List of argument strings used by the contained Args.
    ArgStringList ArgStrings;

    /// The full list of arguments.
    arglist_type Args;

    /// A map of arguments by option ID; in conjunction with the
    /// intrusive list in Arg instances this allows iterating over all
    /// arguments for a particular option.
    llvm::DenseMap<unsigned, Arg*> ArgMap;

  public:
    ArgList(const char **ArgBegin, const char **ArgEnd);
    ArgList(const ArgList &);
    ~ArgList();

    unsigned size() const { return Args.size(); }

    iterator begin() { return Args.begin(); }
    iterator end() { return Args.end(); }

    const_iterator begin() const { return Args.begin(); }
    const_iterator end() const { return Args.end(); }

    Arg *getArgForID(unsigned ID) const { 
      llvm::DenseMap<unsigned, Arg*>::iterator it = ArgMap.find(ID);
      if (it != ArgMap.end())
        return it->second;
      return 0;
    }
  };
} // end namespace driver
} // end namespace clang

#endif
