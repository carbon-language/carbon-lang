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
%option nodefault
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
handleContext(const char* tokenText, llvm::ConfigLexerTokens token) {
  if (ConfigLexerState.in_value) {
    ConfigLexerState.StringVal = tokenText;
    return OPTION;
  }
  return token;
}

inline llvm::ConfigLexerTokens 
handleSubstitution(llvm::ConfigLexerTokens token) {
  if (ConfigLexerState.in_value) 
    return token;
  YY_FATAL_ERROR("Substitition tokens not allowed in names" ); 
  return ERRORTOK;
};

inline llvm::ConfigLexerTokens handleBoolean(llvm::ConfigLexerTokens token) {
  if (ConfigLexerState.in_value)
    return token;
  YY_FATAL_ERROR("Boolean values not allowed in names");
  return ERRORTOK;
}

%}

ASSEMBLER       assembler|Assembler|ASSEMBLER
BadSubst        \%[^iots][a-zA-Z]\%
COMMAND         command|Command|COMMAND
Comment         \#[^\r\n]*\r?\n
NewLine         \r?\n
Eq              \=
EscNewLine      \\\r?\n
GROKS_DASH_O    groks_dash_O|Groks_Dash_O|GROKS_DASH_O
LANG            lang|Lang|LANG
LINKER          linker|Linker|LINKER
NAME            name|Name|NAME
OPT1            opt1|Opt1|OPT1
OPT2            opt2|Opt2|OPT2
OPT3            opt3|Opt3|OPT3
OPT4            opt4|Opt4|OPT4
OPT5            opt5|Opt5|OPT5
OPTIMIZER       optimizer|Optimizer|OPTIMIZER
OPTIMIZES       optimizes|Optimizes|OPTIMIZES
Option          [-A-Za-z0-9_:%+/\\|,][-A-Za-z0-9_:+/\\|,@]*
OUTPUT_IS_ASM   output_is_asm|Output_Is_Asm|OUTPUT_IS_ASM
PREPROCESSES    preprocesses|PreProcesses|PREPROCESSES
PREPROCESSOR    preprocessor|PreProcessor|PREPROCESSOR
REQUIRED        required|Required|REQUIRED
Sep             \.
String          \"[^\"]*\"
TRANSLATES      translates|Translates|TRANSLATES
TRANSLATOR      translator|Translator|TRANSLATOR
VERSION         version|Version|VERSION
White           [ \t]*

True            true|True|TRUE
False           false|False|FALSE
On              on|On|ON
Off             off|Off|OFF
Yes             yes|Yes|YES
No              no|No|NO

%%

{White}         { /* Ignore whitespace */ }

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

{LANG}          { return handleContext("lang",LANG); }
{PREPROCESSOR}  { return handleContext("preprocessor",PREPROCESSOR); }
{TRANSLATOR}    { return handleContext("translator",TRANSLATOR); }
{OPTIMIZER}     { return handleContext("optimizer",OPTIMIZER); }
{ASSEMBLER}     { return handleContext("assembler",ASSEMBLER); }
{LINKER}        { return handleContext("linker",LINKER); }
{NAME}          { return handleContext("name",NAME); }
{REQUIRED}      { return handleContext("required",REQUIRED); }
{COMMAND}       { return handleContext("command",COMMAND); }
{PREPROCESSES}  { return handleContext("preprocesses",PREPROCESSES); }
{TRANSLATES}    { return handleContext("translates",TRANSLATES); }
{OPTIMIZES}     { return handleContext("optimizes",OPTIMIZES); }
{GROKS_DASH_O}  { return handleContext("groks_dash_O",GROKS_DASH_O); }
{OUTPUT_IS_ASM} { return handleContext("output_ias_asm",OUTPUT_IS_ASM); }
{OPT1}          { return handleContext("opt1",OPT1); }
{OPT2}          { return handleContext("opt2",OPT2); }
{OPT3}          { return handleContext("opt3",OPT3); }
{OPT4}          { return handleContext("opt4",OPT4); }
{OPT5}          { return handleContext("opt5",OPT5); }
{VERSION}       { return handleContext("version",VERSION); }

%in%            { return handleSubstitution(IN_SUBST); }
%out%           { return handleSubstitution(OUT_SUBST); }
%time%          { return handleSubstitution(TIME_SUBST); }
%stats%         { return handleSubstitution(STATS_SUBST); }
%opt%           { return handleSubstitution(OPT_SUBST); }
%target%        { return handleSubstitution(TARGET_SUBST); }
{BadSubst}      { YY_FATAL_ERROR("Invalid substitution token"); }
{True}          { return handleBoolean(TRUETOK); }
{On}            { return handleBoolean(TRUETOK); }
{Yes}           { return handleBoolean(TRUETOK); }
{False}         { return handleBoolean(FALSETOK); }
{Off}           { return handleBoolean(FALSETOK); }
{No}            { return handleBoolean(FALSETOK); }

{Option}        { ConfigLexerState.StringVal = yytext; return OPTION; }
{String}        { ConfigLexerState.StringVal = yytext+1;  // Nuke start quote
                  ConfigLexerState.StringVal.erase(
                  --ConfigLexerState.StringVal.end());
                  return STRING;
                }
{Sep}           { if (ConfigLexerState.in_value) { ConfigLexerState.StringVal = yytext;
                    return OPTION; } }


%%
