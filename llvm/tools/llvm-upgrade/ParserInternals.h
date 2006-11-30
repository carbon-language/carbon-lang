//===-- ParserInternals.h - Definitions internal to the parser --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file defines the variables that are shared between the lexer,
//  the parser, and the main program.
//
//===----------------------------------------------------------------------===//

#ifndef PARSER_INTERNALS_H
#define PARSER_INTERNALS_H

#include <string>

// Global variables exported from the lexer...

extern std::string CurFileName;
extern std::string Textin;
extern int Upgradelineno;

// functions exported from the lexer
void set_scan_bytes (const char * str, size_t len);

void UpgradeAssembly(const std::string & infile, std::ostream &out);

// Globals exported by the parser...
extern char* Upgradetext;
extern int   Upgradeleng;

int yyerror(const char *ErrorMsg) ;

#endif
