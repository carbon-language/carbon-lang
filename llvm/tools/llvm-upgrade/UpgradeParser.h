/* A Bison parser, made by GNU Bison 2.1.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005 Free Software Foundation, Inc.

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
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

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
     ESINT64VAL = 258,
     EUINT64VAL = 259,
     SINTVAL = 260,
     UINTVAL = 261,
     FPVAL = 262,
     VOID = 263,
     BOOL = 264,
     SBYTE = 265,
     UBYTE = 266,
     SHORT = 267,
     USHORT = 268,
     INT = 269,
     UINT = 270,
     LONG = 271,
     ULONG = 272,
     FLOAT = 273,
     DOUBLE = 274,
     TYPE = 275,
     LABEL = 276,
     VAR_ID = 277,
     LABELSTR = 278,
     STRINGCONSTANT = 279,
     IMPLEMENTATION = 280,
     ZEROINITIALIZER = 281,
     TRUETOK = 282,
     FALSETOK = 283,
     BEGINTOK = 284,
     ENDTOK = 285,
     DECLARE = 286,
     GLOBAL = 287,
     CONSTANT = 288,
     SECTION = 289,
     VOLATILE = 290,
     TO = 291,
     DOTDOTDOT = 292,
     NULL_TOK = 293,
     UNDEF = 294,
     CONST = 295,
     INTERNAL = 296,
     LINKONCE = 297,
     WEAK = 298,
     APPENDING = 299,
     DLLIMPORT = 300,
     DLLEXPORT = 301,
     EXTERN_WEAK = 302,
     OPAQUE = 303,
     NOT = 304,
     EXTERNAL = 305,
     TARGET = 306,
     TRIPLE = 307,
     ENDIAN = 308,
     POINTERSIZE = 309,
     LITTLE = 310,
     BIG = 311,
     ALIGN = 312,
     DEPLIBS = 313,
     CALL = 314,
     TAIL = 315,
     ASM_TOK = 316,
     MODULE = 317,
     SIDEEFFECT = 318,
     CC_TOK = 319,
     CCC_TOK = 320,
     CSRETCC_TOK = 321,
     FASTCC_TOK = 322,
     COLDCC_TOK = 323,
     X86_STDCALLCC_TOK = 324,
     X86_FASTCALLCC_TOK = 325,
     DATALAYOUT = 326,
     RET = 327,
     BR = 328,
     SWITCH = 329,
     INVOKE = 330,
     UNWIND = 331,
     UNREACHABLE = 332,
     ADD = 333,
     SUB = 334,
     MUL = 335,
     UDIV = 336,
     SDIV = 337,
     FDIV = 338,
     UREM = 339,
     SREM = 340,
     FREM = 341,
     AND = 342,
     OR = 343,
     XOR = 344,
     SETLE = 345,
     SETGE = 346,
     SETLT = 347,
     SETGT = 348,
     SETEQ = 349,
     SETNE = 350,
     MALLOC = 351,
     ALLOCA = 352,
     FREE = 353,
     LOAD = 354,
     STORE = 355,
     GETELEMENTPTR = 356,
     TRUNC = 357,
     ZEXT = 358,
     SEXT = 359,
     FPTRUNC = 360,
     FPEXT = 361,
     BITCAST = 362,
     UITOFP = 363,
     SITOFP = 364,
     FPTOUI = 365,
     FPTOSI = 366,
     INTTOPTR = 367,
     PTRTOINT = 368,
     PHI_TOK = 369,
     SELECT = 370,
     SHL = 371,
     LSHR = 372,
     ASHR = 373,
     VAARG = 374,
     EXTRACTELEMENT = 375,
     INSERTELEMENT = 376,
     SHUFFLEVECTOR = 377,
     CAST = 378
   };
#endif
/* Tokens.  */
#define ESINT64VAL 258
#define EUINT64VAL 259
#define SINTVAL 260
#define UINTVAL 261
#define FPVAL 262
#define VOID 263
#define BOOL 264
#define SBYTE 265
#define UBYTE 266
#define SHORT 267
#define USHORT 268
#define INT 269
#define UINT 270
#define LONG 271
#define ULONG 272
#define FLOAT 273
#define DOUBLE 274
#define TYPE 275
#define LABEL 276
#define VAR_ID 277
#define LABELSTR 278
#define STRINGCONSTANT 279
#define IMPLEMENTATION 280
#define ZEROINITIALIZER 281
#define TRUETOK 282
#define FALSETOK 283
#define BEGINTOK 284
#define ENDTOK 285
#define DECLARE 286
#define GLOBAL 287
#define CONSTANT 288
#define SECTION 289
#define VOLATILE 290
#define TO 291
#define DOTDOTDOT 292
#define NULL_TOK 293
#define UNDEF 294
#define CONST 295
#define INTERNAL 296
#define LINKONCE 297
#define WEAK 298
#define APPENDING 299
#define DLLIMPORT 300
#define DLLEXPORT 301
#define EXTERN_WEAK 302
#define OPAQUE 303
#define NOT 304
#define EXTERNAL 305
#define TARGET 306
#define TRIPLE 307
#define ENDIAN 308
#define POINTERSIZE 309
#define LITTLE 310
#define BIG 311
#define ALIGN 312
#define DEPLIBS 313
#define CALL 314
#define TAIL 315
#define ASM_TOK 316
#define MODULE 317
#define SIDEEFFECT 318
#define CC_TOK 319
#define CCC_TOK 320
#define CSRETCC_TOK 321
#define FASTCC_TOK 322
#define COLDCC_TOK 323
#define X86_STDCALLCC_TOK 324
#define X86_FASTCALLCC_TOK 325
#define DATALAYOUT 326
#define RET 327
#define BR 328
#define SWITCH 329
#define INVOKE 330
#define UNWIND 331
#define UNREACHABLE 332
#define ADD 333
#define SUB 334
#define MUL 335
#define UDIV 336
#define SDIV 337
#define FDIV 338
#define UREM 339
#define SREM 340
#define FREM 341
#define AND 342
#define OR 343
#define XOR 344
#define SETLE 345
#define SETGE 346
#define SETLT 347
#define SETGT 348
#define SETEQ 349
#define SETNE 350
#define MALLOC 351
#define ALLOCA 352
#define FREE 353
#define LOAD 354
#define STORE 355
#define GETELEMENTPTR 356
#define TRUNC 357
#define ZEXT 358
#define SEXT 359
#define FPTRUNC 360
#define FPEXT 361
#define BITCAST 362
#define UITOFP 363
#define SITOFP 364
#define FPTOUI 365
#define FPTOSI 366
#define INTTOPTR 367
#define PTRTOINT 368
#define PHI_TOK 369
#define SELECT 370
#define SHL 371
#define LSHR 372
#define ASHR 373
#define VAARG 374
#define EXTRACTELEMENT 375
#define INSERTELEMENT 376
#define SHUFFLEVECTOR 377
#define CAST 378




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE Upgradelval;



