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
};

extern ConfigLexerInfo ConfigLexerData;
extern unsigned ConfigLexerLine;

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
  LANG,         ///< The item "lang" (and case variants)
  PREPROCESSOR, ///< The item "preprocessor" (and case variants)
  TRANSLATOR,   ///< The item "translator" (and case variants)
  OPTIMIZER,    ///< The item "optimizer" (and case variants)
  ASSEMBLER,    ///< The item "assembler" (and case variants)
  LINKER,       ///< The item "linker" (and case variants)
  NAME,         ///< The item "name" (and case variants)
  NEEDED,       ///< The item "needed" (and case variants)
  COMMAND,      ///< The item "command" (and case variants)
  PREPROCESSES, ///< The item "preprocesses" (and case variants)
  GROKS_DASH_O, ///< The item "groks_dash_O" (and case variants)
  OPTIMIZES,    ///< The item "optimizes" (and case variants)
};

}

#endif
