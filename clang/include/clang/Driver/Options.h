//===--- Options.h - Option info & table ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_OPTIONS_H_
#define CLANG_DRIVER_OPTIONS_H_

namespace clang {
namespace driver {
namespace options {
  enum ID {
    OPT_INVALID = 0, // This is not an option ID.
    OPT_INPUT,       // Reserved ID for input option.
    OPT_UNKNOWN,     // Reserved ID for unknown option.
#define OPTION(NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR) OPT_##ID,
#include "clang/Driver/Options.def"
    LastOption
#undef OPTION
  };
}

  class Arg;
  class InputArgList;
  class Option;

  /// OptTable - Provide access to the Option info table.
  ///
  /// The OptTable class provides a layer of indirection which allows
  /// Option instance to be created lazily. In the common case, only a
  /// few options will be needed at runtime; the OptTable class
  /// maintains enough information to parse command lines without
  /// instantiating Options, while letting other parts of the driver
  /// still use Option instances where convient.
  class OptTable {
    /// The table of options which have been constructed, indexed by
    /// option::ID - 1.
    mutable Option **Options;

    /// The index of the first option which can be parsed (i.e., is
    /// not a special option like 'input' or 'unknown', and is not an
    /// option group).
    unsigned FirstSearchableOption;

    Option *constructOption(options::ID id) const;

  public:
    OptTable();
    ~OptTable();

    unsigned getNumOptions() const;

    const char *getOptionName(options::ID id) const;

    /// getOption - Get the given \arg id's Option instance, lazily
    /// creating it if necessary.
    const Option *getOption(options::ID id) const;

    /// getOptionKind - Get the kind of the given option.
    unsigned getOptionKind(options::ID id) const;

    /// getOptionHelpText - Get the help text to use to describe this
    /// option.
    const char *getOptionHelpText(options::ID id) const;

    /// getOptionMetaVar - Get the meta-variable name to use when
    /// describing this options values in the help text.
    const char *getOptionMetaVar(options::ID id) const;

    /// parseOneArg - Parse a single argument; returning the new
    /// argument and updating Index.
    ///
    /// \param [in] [out] Index - The current parsing position in the
    /// argument string list; on return this will be the index of the
    /// next argument string to parse.
    ///
    /// \return - The parsed argument, or 0 if the argument is missing
    /// values (in which case Index still points at the conceptual
    /// next argument string to parse).
    Arg *ParseOneArg(const InputArgList &Args, unsigned &Index) const;
  };
}
}

#endif
