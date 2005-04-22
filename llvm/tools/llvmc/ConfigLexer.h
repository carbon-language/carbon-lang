//===- ConfigLexer.h - ConfigLexer Declarations -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the types and data needed by ConfigLexer.l
//
//===------------------------------------------------------------------------===
#ifndef LLVM_TOOLS_LLVMC_CONFIGLEXER_H
#define LLVM_TOOLS_LLVMC_CONFIGLEXER_H

#include <string>
#include <istream>
#include <cassert>

namespace llvm {

struct ConfigLexerInfo
{
  int64_t     IntegerVal;
  std::string StringVal;
  bool in_value;
  unsigned lineNum;
};

extern ConfigLexerInfo ConfigLexerState;

class InputProvider {
  public:
    InputProvider(const std::string& nm) {
      name = nm;
      errCount = 0;
    }
    virtual ~InputProvider();
    virtual unsigned read(char *buf, unsigned max_size) = 0;
    virtual void error(const std::string& msg);
    virtual void checkErrors();

  private:
    std::string name;
    unsigned errCount;
};

extern InputProvider* ConfigLexerInput;

enum ConfigLexerTokens {
  EOFTOK = 0,   ///< Returned by Configlex when we hit end of file
  EOLTOK,       ///< End of line
  ERRORTOK,     ///< Error token
  ARGS_SUBST,   ///< THe substitution item %args%
  ASSEMBLY,     ///< The value "assembly" (and variants)
  ASSEMBLER,    ///< The name "assembler" (and variants)
  BYTECODE,     ///< The value "bytecode" (and variants)
  COMMAND,      ///< The name "command" (and variants)
  DEFS_SUBST,   ///< The substitution item %defs%
  EQUALS,       ///< The equals sign, =
  FALSETOK,     ///< A boolean false value (false/no/off)
  FOPTS_SUBST,  ///< The substitution item %fOpts%
  IN_SUBST,     ///< The substitution item %in%
  INCLS_SUBST,  ///< The substitution item %incls%
  INTEGER,      ///< An integer
  LANG,         ///< The name "lang" (and variants)
  LIBPATHS,     ///< The name "libpaths" (and variants)
  LIBS,         ///< The name "libs" (and variants)
  LIBS_SUBST,   ///< The substitution item %libs%
  LINKER,       ///< The name "linker" (and variants)
  MOPTS_SUBST,  ///< The substitution item %Mopts%
  NAME,         ///< The name "name" (and variants)
  OPT_SUBST,    ///< The substitution item %opt%
  OPTIMIZER,    ///< The name "optimizer" (and variants)
  OPTION,       ///< A command line option
  OPT1,         ///< The name "opt1" (and variants)
  OPT2,         ///< The name "opt2" (and variants)
  OPT3,         ///< The name "opt3" (and variants)
  OPT4,         ///< The name "opt4" (and variants)
  OPT5,         ///< The name "opt5" (and variants)
  OUT_SUBST,    ///< The output substitution item %out%
  OUTPUT,       ///< The name "output" (and variants)
  PREPROCESSES, ///< The name "preprocesses" (and variants)
  PREPROCESSOR, ///< The name "preprocessor" (and variants)
  REQUIRED,     ///< The name "required" (and variants)
  SEPARATOR,    ///< A configuration item separator
  SPACE,        ///< Space between options
  STATS_SUBST,  ///< The stats substitution item %stats%
  STRING,       ///< A quoted string
  TARGET_SUBST, ///< The substitition item %target%
  TIME_SUBST,   ///< The substitution item %time%
  TRANSLATES,   ///< The name "translates" (and variants)
  TRANSLATOR,   ///< The name "translator" (and variants)
  TRUETOK,      ///< A boolean true value (true/yes/on)
  VERBOSE_SUBST,///< The substitution item %verbose%
  VERSION_TOK,  ///< The name "version" (and variants)
  WOPTS_SUBST,  ///< The %WOpts% substitution
};

extern ConfigLexerTokens Configlex();

}

#endif
