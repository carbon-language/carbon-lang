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
NEEDED          needed|Needed|NEEDED
COMMAND         command|Command|COMMAND
PREPROCESSES    preprocesses|PreProcesses|PREPROCESSES
GROKS_DASH_O    groks_dash_O|Groks_Dash_O|GROKS_DASH_O
OPTIMIZES       optimizes|Optimizes|OPTIMIZES
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

{NewLine}       { in_value = false; ConfigLexerLine++; return EOLTOK; }
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
{NEEDED}        { if (in_value) { ConfigLexerData.StringVal = "needed";
                    return OPTION; } else return NEEDED; }
{COMMAND}       { if (in_value) { ConfigLexerData.StringVal = "command";
                    return OPTION; } else return COMMAND; }
{PREPROCESSES}  { if (in_value) { ConfigLexerData.StringVal = "preprocesses";
                    return OPTION; } else return PREPROCESSES; }
{GROKS_DASH_O}  { if (in_value) { ConfigLexerData.StringVal = "groks_dash_O";
                    return OPTION; } else return GROKS_DASH_O; }
{OPTIMIZES}     { if (in_value) { ConfigLexerData.StringVal = "optimizes";
                    return OPTION; } else return OPTIMIZES; }
{Sep}           { if (in_value) { ConfigLexerData.StringVal = yytext;
                    return OPTION; } }

@in@            { if (in_value) return IN_SUBST; else return ERRORTOK;  }
@out@           { if (in_value) return OUT_SUBST; else return ERRORTOK; }
{True}          { if (in_value) return TRUETOK; else return ERRORTOK; }
{On}            { if (in_value) return TRUETOK; else return ERRORTOK; }
{Yes}           { if (in_value) return TRUETOK; else return ERRORTOK; }
{False}         { if (in_value) return FALSETOK; else return ERRORTOK; }
{Off}           { if (in_value) return FALSETOK; else return ERRORTOK; }
{No}            { if (in_value) return FALSETOK; else return ERRORTOK; }

{Eq}            { in_value = true; return EQUALS; }
{Option}        { ConfigLexerData.StringVal = yytext; return OPTION; }
{Integer}       { ConfigLexerData.IntegerVal = IntToVal(yytext); return INTEGER; }
{String}        { yytext[yyleng-1] = 0;          // nuke end quote
                  ConfigLexerData.StringVal = yytext+1;  // Nuke start quote
                  return STRING;
                }

%%
