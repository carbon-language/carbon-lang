/*===- ConfigLexer.l - Scanner for CompilerDriver Config Files -*- C++ -*--===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the flex scanner for configuration files for the
// llvmc CompilerDriver.
//
//===----------------------------------------------------------------------===*/


%option prefix="Config"
%option nostdinit
%option never-interactive
%option batch
%option noyywrap
%option 8bit
%option outfile="ConfigLexer.cpp"
%option ecs
%option noyymore
%option noreject
%pointer

%{

#include "ConfigLexer.h"

#define YY_INPUT(buf,result,max_size) \
  { \
    assert(ConfigLexerInput != 0 && "Oops"); \
    result = ConfigLexerInput->read(buf,max_size); \
    if (result == 0 ) result = YY_NULL; \
  }

#define YY_FATAL_ERROR(msg) \
  { \
    assert(ConfigLexerInput != 0 && "Oops"); \
    ConfigLexerInput->error(msg); \
  }

#define YY_DECL ConfigLexerTokens llvm::Configlex()

#define yyterminate() { return EOFTOK; }

using namespace llvm;

inline llvm::ConfigLexerTokens 
handleNameContext(llvm::ConfigLexerTokens token) {
  ConfigLexerState.StringVal = yytext;
  if (ConfigLexerState.in_value)
    return OPTION;
  return token;
}

inline llvm::ConfigLexerTokens 
handleSubstitution(llvm::ConfigLexerTokens token) {
  if (ConfigLexerState.in_value) {
    ConfigLexerState.StringVal = yytext;
    return token;
  }
  YY_FATAL_ERROR("Substitition tokens not allowed in names" ); 
  return ERRORTOK;
};

inline llvm::ConfigLexerTokens handleValueContext(llvm::ConfigLexerTokens token) {
  ConfigLexerState.StringVal = yytext;
  if (ConfigLexerState.in_value)
    return token;
  return OPTION;
}

%}

ASSEMBLER       assembler|Assembler|ASSEMBLER
COMMAND         command|Command|COMMAND
LANG            lang|Lang|LANG
LIBS            libs|Libs|LIBS
LINKER		linker|Linker|LINKER
NAME            name|Name|NAME
OPT1            opt1|Opt1|OPT1
OPT2            opt2|Opt2|OPT2
OPT3            opt3|Opt3|OPT3
OPT4            opt4|Opt4|OPT4
OPT5            opt5|Opt5|OPT5
OPTIMIZER       optimizer|Optimizer|OPTIMIZER
OUTPUT          output|Output|OUTPUT
PREPROCESSES    preprocesses|PreProcesses|PREPROCESSES
PREPROCESSOR    preprocessor|PreProcessor|PREPROCESSOR
REQUIRED        required|Required|REQUIRED
TRANSLATES      translates|Translates|TRANSLATES
TRANSLATOR      translator|Translator|TRANSLATOR
VERSION         version|Version|VERSION

True            true|True|TRUE|on|On|ON|yes|Yes|YES
False           false|False|FALSE|off|Off|OFF|no|No|NO
Bytecode        bc|BC|bytecode|Bytecode|BYTECODE
Assembly        asm|ASM|assembly|Assembly|ASSEMBLY

BadSubst        \%[a-zA-Z]*\%
Comment         \#[^\r\n]*\r?\n
NewLine         \r?\n
Eq              \=
EscNewLine      \\\r?\n
Option          [-A-Za-z0-9_:%+/\\|,][-A-Za-z0-9_:+/\\|,@]*
Sep             \.
String          \"[^\"]*\"
White           [ \t]*


%%

{White}         { if (ConfigLexerState.in_value) return SPACE; }

{Comment}       { /* Ignore comments */
                  ConfigLexerState.in_value = false; 
                  ConfigLexerState.lineNum++; 
                  return EOLTOK; 
                }

{EscNewLine}    { ConfigLexerState.lineNum++; 
                  /* Don't return EOLTOK! */
                }

{NewLine}       { ConfigLexerState.in_value = false; 
                  ConfigLexerState.lineNum++; 
                  return EOLTOK; 
                }

{Eq}            { ConfigLexerState.in_value = true; 
                  return EQUALS; 
                }

{Sep}           { return SEPARATOR; }

{VERSION}       { return handleNameContext(VERSION_TOK); }

{LANG}          { return handleNameContext(LANG); }
{LIBS}          { return handleNameContext(LIBS); }
{NAME}          { return handleNameContext(NAME); }
{OPT1}          { return handleNameContext(OPT1); }
{OPT2}          { return handleNameContext(OPT2); }
{OPT3}          { return handleNameContext(OPT3); }
{OPT4}          { return handleNameContext(OPT4); }
{OPT5}          { return handleNameContext(OPT5); }

{PREPROCESSOR}  { return handleNameContext(PREPROCESSOR); }
{COMMAND}       { return handleNameContext(COMMAND); }
{REQUIRED}      { return handleNameContext(REQUIRED); }

{TRANSLATOR}    { return handleNameContext(TRANSLATOR); }
{PREPROCESSES}  { return handleNameContext(PREPROCESSES); }
{OUTPUT}        { return handleNameContext(OUTPUT); }

{OPTIMIZER}     { return handleNameContext(OPTIMIZER); }
{TRANSLATES}    { return handleNameContext(TRANSLATES); }

{ASSEMBLER}     { return handleNameContext(ASSEMBLER); }

{LINKER}        { return handleNameContext(LINKER); }

%args%          { return handleSubstitution(ARGS_SUBST); }
%defs%          { return handleSubstitution(DEFS_SUBST); }
%in%            { return handleSubstitution(IN_SUBST); }
%incls%         { return handleSubstitution(INCLS_SUBST); }
%libs%          { return handleSubstitution(LIBS_SUBST); }
%opt%           { return handleSubstitution(OPT_SUBST); }
%out%           { return handleSubstitution(OUT_SUBST); }
%stats%         { return handleSubstitution(STATS_SUBST); }
%target%        { return handleSubstitution(TARGET_SUBST); }
%time%          { return handleSubstitution(TIME_SUBST); }
%verbose%       { return handleSubstitution(VERBOSE_SUBST); }
%fOpts%         { return handleSubstitution(FOPTS_SUBST); }
%MOpts%         { return handleSubstitution(MOPTS_SUBST); }
%WOpts%         { return handleSubstitution(WOPTS_SUBST); }

{Assembly}      { return handleValueContext(ASSEMBLY); }
{Bytecode}      { return handleValueContext(BYTECODE); }
{True}          { return handleValueContext(TRUETOK); }
{False}         { return handleValueContext(FALSETOK); }

{Option}        { ConfigLexerState.StringVal = yytext; return OPTION; }
{String}        { ConfigLexerState.StringVal = yytext+1;  // Nuke start quote
                  ConfigLexerState.StringVal.erase(
                  --ConfigLexerState.StringVal.end());
                  return STRING;
                }
{BadSubst}      { YY_FATAL_ERROR("Invalid substitution token"); }

%%
