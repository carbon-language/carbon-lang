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
%option yylineno
%option nostdinit
%option never-interactive
%option batch
%option noyywrap
%option nodefault
%option 8bit
%option outfile="ConfigLexer.cpp"
%option ecs
%option noreject
%option noyymore
%array

%{

#include "ConfigLexer.h"

#define YY_INPUT(buf,result,max_size) \
  { \
    assert(ConfigLexerInput != 0 && "Oops"); \
    result = ConfigLexerInput->read(buf,max_size); \
    if (result == 0 ) result = YY_NULL; \
  }

#define YY_DECL ConfigLexerTokens llvm::Configlex()

#define yyterminate() { return EOFTOK; }

using namespace llvm;

/* Conversion of text ints to binary */
static int64_t IntToVal(const char *Buffer) {
  int64_t Result = 0;
  for (; *Buffer; Buffer++) {
    int64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
  }
  return Result;
}

bool in_value = false;

%}

LANG            lang|Lang|LANG
PREPROCESSOR    preprocessor|PreProcessor|PREPROCESSOR
TRANSLATOR      translator|Translator|TRANSLATOR
OPTIMIZER       optimizer|Optimizer|OPTIMIZER
ASSEMBLER       assembler|Assembler|ASSEMBLER
LINKER          linker|Linker|LINKER
NAME            name|Name|NAME
REQUIRED        required|Required|REQUIRED
COMMAND         command|Command|COMMAND
PREPROCESSES    preprocesses|PreProcesses|PREPROCESSES
GROKS_DASH_O    groks_dash_O|Groks_Dash_O|GROKS_DASH_O
GROKS_O10N      groks_optimization|Groks_Optimization|GROKS_OPTIMIZATION
OPTIMIZES       optimizes|Optimizes|OPTIMIZES
OPT1            opt1|Opt1|OPT1
OPT2            opt2|Opt2|OPT2
OPT3            opt3|Opt3|OPT3
OPT4            opt4|Opt4|OPT4
OPT5            opt5|Opt5|OPT5
Comment         \#[^\n]*
NewLine         \n
White           [ \t]*
Option          [-A-Za-z0-9_:%+/\\|,]*
Sep             \.
Eq              \=
String          \"[^\"]*\"
Integer         [-+]?[0-9]+
True            true|True|TRUE
False           false|False|FALSE
On              on|On|ON
Off             off|Off|OFF
Yes             yes|Yes|YES
No              no|No|NO

%%

{NewLine}       { in_value = false; return EOLTOK; }
{Eq}            { in_value = true; return EQUALS; }
{Comment}       { /* Ignore comments */ }
{White}         { /* Ignore whitespace */ }

{LANG}          { if (in_value) { ConfigLexerData.StringVal = "lang"; 
                    return OPTION;  } else return LANG; }
{PREPROCESSOR}  { if (in_value) { ConfigLexerData.StringVal = "preprocessor";
                    return OPTION; } else return PREPROCESSOR; }
{TRANSLATOR}    { if (in_value) { ConfigLexerData.StringVal = "translator";
                    return OPTION; } else return TRANSLATOR; }
{OPTIMIZER}     { if (in_value) { ConfigLexerData.StringVal = "optimizer";
                    return OPTION; } else return OPTIMIZER; }
{ASSEMBLER}     { if (in_value) { ConfigLexerData.StringVal = "assembler";
                    return OPTION; } else return ASSEMBLER; }
{LINKER}        { if (in_value) { ConfigLexerData.StringVal = "linker";
                    return OPTION; } else return LINKER; }
{NAME}          { if (in_value) { ConfigLexerData.StringVal = "name";
                    return OPTION; } else return NAME; }
{REQUIRED}      { if (in_value) { ConfigLexerData.StringVal = "required";
                    return OPTION; } else return REQUIRED; }
{COMMAND}       { if (in_value) { ConfigLexerData.StringVal = "command";
                    return OPTION; } else return COMMAND; }
{PREPROCESSES}  { if (in_value) { ConfigLexerData.StringVal = "preprocesses";
                    return OPTION; } else return PREPROCESSES; }
{GROKS_DASH_O}  { if (in_value) { ConfigLexerData.StringVal = "groks_dash_O";
                    return OPTION; } else return GROKS_DASH_O; }
{GROKS_O10N}    { if (in_value) { ConfigLexerData.StringVal = 
                    "groks_optimization"; return OPTION; } 
                    else return GROKS_O10N; }
{OPTIMIZES}     { if (in_value) { ConfigLexerData.StringVal = "optimizes";
                    return OPTION; } else return OPTIMIZES; }
{OPT1}          { if (in_value) { ConfigLexerData.StringVal = "opt1";
                    return OPTION; } else return OPT1; }
{OPT2}          { if (in_value) { ConfigLexerData.StringVal = "opt2";
                    return OPTION; } else return OPT2; }
{OPT3}          { if (in_value) { ConfigLexerData.StringVal = "opt3";
                    return OPTION; } else return OPT3; }
{OPT4}          { if (in_value) { ConfigLexerData.StringVal = "opt4";
                    return OPTION; } else return OPT4; }
{OPT5}          { if (in_value) { ConfigLexerData.StringVal = "opt5";
                    return OPTION; } else return OPT5; }
@in@            { if (in_value) return IN_SUBST; else return ERRORTOK;  }
@out@           { if (in_value) return OUT_SUBST; else return ERRORTOK; }
{True}          { if (in_value) return TRUETOK; else return ERRORTOK; }
{On}            { if (in_value) return TRUETOK; else return ERRORTOK; }
{Yes}           { if (in_value) return TRUETOK; else return ERRORTOK; }
{False}         { if (in_value) return FALSETOK; else return ERRORTOK; }
{Off}           { if (in_value) return FALSETOK; else return ERRORTOK; }
{No}            { if (in_value) return FALSETOK; else return ERRORTOK; }

{Option}        { ConfigLexerData.StringVal = yytext; return OPTION; }
{Integer}       { ConfigLexerData.IntegerVal = IntToVal(yytext); return INTEGER; }
{String}        { yytext[yyleng-1] = 0;          // nuke end quote
                  ConfigLexerData.StringVal = yytext+1;  // Nuke start quote
                  return STRING;
                }
{Sep}           { if (in_value) { ConfigLexerData.StringVal = yytext;
                    return OPTION; } }


%%
