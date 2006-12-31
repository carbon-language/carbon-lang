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
     VOID = 258,
     BOOL = 259,
     SBYTE = 260,
     UBYTE = 261,
     SHORT = 262,
     USHORT = 263,
     INT = 264,
     UINT = 265,
     LONG = 266,
     ULONG = 267,
     FLOAT = 268,
     DOUBLE = 269,
     LABEL = 270,
     OPAQUE = 271,
     ESINT64VAL = 272,
     EUINT64VAL = 273,
     SINTVAL = 274,
     UINTVAL = 275,
     FPVAL = 276,
     NULL_TOK = 277,
     UNDEF = 278,
     ZEROINITIALIZER = 279,
     TRUETOK = 280,
     FALSETOK = 281,
     TYPE = 282,
     VAR_ID = 283,
     LABELSTR = 284,
     STRINGCONSTANT = 285,
     IMPLEMENTATION = 286,
     BEGINTOK = 287,
     ENDTOK = 288,
     DECLARE = 289,
     GLOBAL = 290,
     CONSTANT = 291,
     SECTION = 292,
     VOLATILE = 293,
     TO = 294,
     DOTDOTDOT = 295,
     CONST = 296,
     INTERNAL = 297,
     LINKONCE = 298,
     WEAK = 299,
     DLLIMPORT = 300,
     DLLEXPORT = 301,
     EXTERN_WEAK = 302,
     APPENDING = 303,
     NOT = 304,
     EXTERNAL = 305,
     TARGET = 306,
     TRIPLE = 307,
     ENDIAN = 308,
     POINTERSIZE = 309,
     LITTLE = 310,
     BIG = 311,
     ALIGN = 312,
     UNINITIALIZED = 313,
     DEPLIBS = 314,
     CALL = 315,
     TAIL = 316,
     ASM_TOK = 317,
     MODULE = 318,
     SIDEEFFECT = 319,
     CC_TOK = 320,
     CCC_TOK = 321,
     CSRETCC_TOK = 322,
     FASTCC_TOK = 323,
     COLDCC_TOK = 324,
     X86_STDCALLCC_TOK = 325,
     X86_FASTCALLCC_TOK = 326,
     DATALAYOUT = 327,
     RET = 328,
     BR = 329,
     SWITCH = 330,
     INVOKE = 331,
     EXCEPT = 332,
     UNWIND = 333,
     UNREACHABLE = 334,
     ADD = 335,
     SUB = 336,
     MUL = 337,
     DIV = 338,
     UDIV = 339,
     SDIV = 340,
     FDIV = 341,
     REM = 342,
     UREM = 343,
     SREM = 344,
     FREM = 345,
     AND = 346,
     OR = 347,
     XOR = 348,
     SETLE = 349,
     SETGE = 350,
     SETLT = 351,
     SETGT = 352,
     SETEQ = 353,
     SETNE = 354,
     ICMP = 355,
     FCMP = 356,
     EQ = 357,
     NE = 358,
     SLT = 359,
     SGT = 360,
     SLE = 361,
     SGE = 362,
     OEQ = 363,
     ONE = 364,
     OLT = 365,
     OGT = 366,
     OLE = 367,
     OGE = 368,
     ORD = 369,
     UNO = 370,
     UEQ = 371,
     UNE = 372,
     ULT = 373,
     UGT = 374,
     ULE = 375,
     UGE = 376,
     MALLOC = 377,
     ALLOCA = 378,
     FREE = 379,
     LOAD = 380,
     STORE = 381,
     GETELEMENTPTR = 382,
     PHI_TOK = 383,
     SELECT = 384,
     SHL = 385,
     SHR = 386,
     ASHR = 387,
     LSHR = 388,
     VAARG = 389,
     EXTRACTELEMENT = 390,
     INSERTELEMENT = 391,
     SHUFFLEVECTOR = 392,
     CAST = 393,
     TRUNC = 394,
     ZEXT = 395,
     SEXT = 396,
     FPTRUNC = 397,
     FPEXT = 398,
     FPTOUI = 399,
     FPTOSI = 400,
     UITOFP = 401,
     SITOFP = 402,
     PTRTOINT = 403,
     INTTOPTR = 404,
     BITCAST = 405
   };
#endif
/* Tokens.  */
#define VOID 258
#define BOOL 259
#define SBYTE 260
#define UBYTE 261
#define SHORT 262
#define USHORT 263
#define INT 264
#define UINT 265
#define LONG 266
#define ULONG 267
#define FLOAT 268
#define DOUBLE 269
#define LABEL 270
#define OPAQUE 271
#define ESINT64VAL 272
#define EUINT64VAL 273
#define SINTVAL 274
#define UINTVAL 275
#define FPVAL 276
#define NULL_TOK 277
#define UNDEF 278
#define ZEROINITIALIZER 279
#define TRUETOK 280
#define FALSETOK 281
#define TYPE 282
#define VAR_ID 283
#define LABELSTR 284
#define STRINGCONSTANT 285
#define IMPLEMENTATION 286
#define BEGINTOK 287
#define ENDTOK 288
#define DECLARE 289
#define GLOBAL 290
#define CONSTANT 291
#define SECTION 292
#define VOLATILE 293
#define TO 294
#define DOTDOTDOT 295
#define CONST 296
#define INTERNAL 297
#define LINKONCE 298
#define WEAK 299
#define DLLIMPORT 300
#define DLLEXPORT 301
#define EXTERN_WEAK 302
#define APPENDING 303
#define NOT 304
#define EXTERNAL 305
#define TARGET 306
#define TRIPLE 307
#define ENDIAN 308
#define POINTERSIZE 309
#define LITTLE 310
#define BIG 311
#define ALIGN 312
#define UNINITIALIZED 313
#define DEPLIBS 314
#define CALL 315
#define TAIL 316
#define ASM_TOK 317
#define MODULE 318
#define SIDEEFFECT 319
#define CC_TOK 320
#define CCC_TOK 321
#define CSRETCC_TOK 322
#define FASTCC_TOK 323
#define COLDCC_TOK 324
#define X86_STDCALLCC_TOK 325
#define X86_FASTCALLCC_TOK 326
#define DATALAYOUT 327
#define RET 328
#define BR 329
#define SWITCH 330
#define INVOKE 331
#define EXCEPT 332
#define UNWIND 333
#define UNREACHABLE 334
#define ADD 335
#define SUB 336
#define MUL 337
#define DIV 338
#define UDIV 339
#define SDIV 340
#define FDIV 341
#define REM 342
#define UREM 343
#define SREM 344
#define FREM 345
#define AND 346
#define OR 347
#define XOR 348
#define SETLE 349
#define SETGE 350
#define SETLT 351
#define SETGT 352
#define SETEQ 353
#define SETNE 354
#define ICMP 355
#define FCMP 356
#define EQ 357
#define NE 358
#define SLT 359
#define SGT 360
#define SLE 361
#define SGE 362
#define OEQ 363
#define ONE 364
#define OLT 365
#define OGT 366
#define OLE 367
#define OGE 368
#define ORD 369
#define UNO 370
#define UEQ 371
#define UNE 372
#define ULT 373
#define UGT 374
#define ULE 375
#define UGE 376
#define MALLOC 377
#define ALLOCA 378
#define FREE 379
#define LOAD 380
#define STORE 381
#define GETELEMENTPTR 382
#define PHI_TOK 383
#define SELECT 384
#define SHL 385
#define SHR 386
#define ASHR 387
#define LSHR 388
#define VAARG 389
#define EXTRACTELEMENT 390
#define INSERTELEMENT 391
#define SHUFFLEVECTOR 392
#define CAST 393
#define TRUNC 394
#define ZEXT 395
#define SEXT 396
#define FPTRUNC 397
#define FPEXT 398
#define FPTOUI 399
#define FPTOSI 400
#define UITOFP 401
#define SITOFP 402
#define PTRTOINT 403
#define INTTOPTR 404
#define BITCAST 405




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 280 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
typedef union YYSTYPE {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
} YYSTYPE;
/* Line 1447 of yacc.c.  */
#line 346 "UpgradeParser.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE Upgradelval;



