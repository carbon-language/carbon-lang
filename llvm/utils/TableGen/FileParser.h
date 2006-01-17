/* A Bison parser, made by GNU Bison 1.875.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     INT = 258,
     BIT = 259,
     STRING = 260,
     BITS = 261,
     LIST = 262,
     CODE = 263,
     DAG = 264,
     CLASS = 265,
     DEF = 266,
     FIELD = 267,
     LET = 268,
     IN = 269,
     SHLTOK = 270,
     SRATOK = 271,
     SRLTOK = 272,
     INTVAL = 273,
     ID = 274,
     VARNAME = 275,
     STRVAL = 276,
     CODEFRAGMENT = 277
   };
#endif
#define INT 258
#define BIT 259
#define STRING 260
#define BITS 261
#define LIST 262
#define CODE 263
#define DAG 264
#define CLASS 265
#define DEF 266
#define FIELD 267
#define LET 268
#define IN 269
#define SHLTOK 270
#define SRATOK 271
#define SRLTOK 272
#define INTVAL 273
#define ID 274
#define VARNAME 275
#define STRVAL 276
#define CODEFRAGMENT 277




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 189 "/home/vadve/criswell/llvm/utils/TableGen/FileParser.y"
typedef union YYSTYPE {
  std::string*                StrVal;
  int                         IntVal;
  llvm::RecTy*                Ty;
  llvm::Init*                 Initializer;
  std::vector<llvm::Init*>*   FieldList;
  std::vector<unsigned>*      BitList;
  llvm::Record*               Rec;
  SubClassRefTy*              SubClassRef;
  std::vector<SubClassRefTy>* SubClassList;
  std::vector<std::pair<llvm::Init*, std::string> >* DagValueList;
} YYSTYPE;
/* Line 1240 of yacc.c.  */
#line 93 "FileParser.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE Filelval;



