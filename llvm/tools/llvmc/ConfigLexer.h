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
  OPTION,       ///< A command line option
  SEPARATOR,    ///< A configuration item separator
  EQUALS,       ///< The equals sign, =
  TRUETOK,      ///< A boolean true value (true/yes/on)
  FALSETOK,     ///< A boolean false value (false/no/off)
  INTEGER,      ///< An integer 
  STRING,       ///< A quoted string
  IN_SUBST,     ///< The input substitution item @in@
  OUT_SUBST,    ///< The output substitution item @out@
  STATS_SUBST,  ///< The stats substitution item @stats@
  TIME_SUBST,   ///< The substitution item @time@
  OPT_SUBST,    ///< The substitution item @opt@
  TARGET_SUBST, ///< The substitition item @target@
  LANG,         ///< The item "lang" (and case variants)
  PREPROCESSOR, ///< The item "preprocessor" (and case variants)
  TRANSLATOR,   ///< The item "translator" (and case variants)
  OPTIMIZER,    ///< The item "optimizer" (and case variants)
  ASSEMBLER,    ///< The item "assembler" (and case variants)
  LINKER,       ///< The item "linker" (and case variants)
  NAME,         ///< The item "name" (and case variants)
  REQUIRED,     ///< The item "required" (and case variants)
  COMMAND,      ///< The item "command" (and case variants)
  PREPROCESSES, ///< The item "preprocesses" (and case variants)
  TRANSLATES,   ///< The item "translates" (and case variants)
  OPTIMIZES,    ///< The item "optimizes" (and case variants)
  GROKS_DASH_O, ///< The item "groks_dash_O" (and case variants)
  OUTPUT_IS_ASM,///< The item "outut_is_asm" (and case variants)
  OPT1,         ///< The item "opt1" (and case variants)
  OPT2,         ///< The item "opt2" (and case variants)
  OPT3,         ///< The item "opt3" (and case variants)
  OPT4,         ///< The item "opt4" (and case variants)
  OPT5,         ///< The item "opt5" (and case variants)
};

extern ConfigLexerTokens Configlex();

}

#endif
