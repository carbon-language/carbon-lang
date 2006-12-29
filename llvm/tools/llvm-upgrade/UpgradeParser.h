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
     DEFINE = 290,
     GLOBAL = 291,
     CONSTANT = 292,
     SECTION = 293,
     VOLATILE = 294,
     TO = 295,
     DOTDOTDOT = 296,
     CONST = 297,
     INTERNAL = 298,
     LINKONCE = 299,
     WEAK = 300,
     DLLIMPORT = 301,
     DLLEXPORT = 302,
     EXTERN_WEAK = 303,
     APPENDING = 304,
     NOT = 305,
     EXTERNAL = 306,
     TARGET = 307,
     TRIPLE = 308,
     ENDIAN = 309,
     POINTERSIZE = 310,
     LITTLE = 311,
     BIG = 312,
     ALIGN = 313,
     UNINITIALIZED = 314,
     DEPLIBS = 315,
     CALL = 316,
     TAIL = 317,
     ASM_TOK = 318,
     MODULE = 319,
     SIDEEFFECT = 320,
     CC_TOK = 321,
     CCC_TOK = 322,
     CSRETCC_TOK = 323,
     FASTCC_TOK = 324,
     COLDCC_TOK = 325,
     X86_STDCALLCC_TOK = 326,
     X86_FASTCALLCC_TOK = 327,
     DATALAYOUT = 328,
     RET = 329,
     BR = 330,
     SWITCH = 331,
     INVOKE = 332,
     EXCEPT = 333,
     UNWIND = 334,
     UNREACHABLE = 335,
     ADD = 336,
     SUB = 337,
     MUL = 338,
     DIV = 339,
     UDIV = 340,
     SDIV = 341,
     FDIV = 342,
     REM = 343,
     UREM = 344,
     SREM = 345,
     FREM = 346,
     AND = 347,
     OR = 348,
     XOR = 349,
     SETLE = 350,
     SETGE = 351,
     SETLT = 352,
     SETGT = 353,
     SETEQ = 354,
     SETNE = 355,
     ICMP = 356,
     FCMP = 357,
     EQ = 358,
     NE = 359,
     SLT = 360,
     SGT = 361,
     SLE = 362,
     SGE = 363,
     OEQ = 364,
     ONE = 365,
     OLT = 366,
     OGT = 367,
     OLE = 368,
     OGE = 369,
     ORD = 370,
     UNO = 371,
     UEQ = 372,
     UNE = 373,
     ULT = 374,
     UGT = 375,
     ULE = 376,
     UGE = 377,
     MALLOC = 378,
     ALLOCA = 379,
     FREE = 380,
     LOAD = 381,
     STORE = 382,
     GETELEMENTPTR = 383,
     PHI_TOK = 384,
     SELECT = 385,
     SHL = 386,
     SHR = 387,
     ASHR = 388,
     LSHR = 389,
     VAARG = 390,
     EXTRACTELEMENT = 391,
     INSERTELEMENT = 392,
     SHUFFLEVECTOR = 393,
     CAST = 394,
     TRUNC = 395,
     ZEXT = 396,
     SEXT = 397,
     FPTRUNC = 398,
     FPEXT = 399,
     FPTOUI = 400,
     FPTOSI = 401,
     UITOFP = 402,
     SITOFP = 403,
     PTRTOINT = 404,
     INTTOPTR = 405,
     BITCAST = 406
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
#define DEFINE 290
#define GLOBAL 291
#define CONSTANT 292
#define SECTION 293
#define VOLATILE 294
#define TO 295
#define DOTDOTDOT 296
#define CONST 297
#define INTERNAL 298
#define LINKONCE 299
#define WEAK 300
#define DLLIMPORT 301
#define DLLEXPORT 302
#define EXTERN_WEAK 303
#define APPENDING 304
#define NOT 305
#define EXTERNAL 306
#define TARGET 307
#define TRIPLE 308
#define ENDIAN 309
#define POINTERSIZE 310
#define LITTLE 311
#define BIG 312
#define ALIGN 313
#define UNINITIALIZED 314
#define DEPLIBS 315
#define CALL 316
#define TAIL 317
#define ASM_TOK 318
#define MODULE 319
#define SIDEEFFECT 320
#define CC_TOK 321
#define CCC_TOK 322
#define CSRETCC_TOK 323
#define FASTCC_TOK 324
#define COLDCC_TOK 325
#define X86_STDCALLCC_TOK 326
#define X86_FASTCALLCC_TOK 327
#define DATALAYOUT 328
#define RET 329
#define BR 330
#define SWITCH 331
#define INVOKE 332
#define EXCEPT 333
#define UNWIND 334
#define UNREACHABLE 335
#define ADD 336
#define SUB 337
#define MUL 338
#define DIV 339
#define UDIV 340
#define SDIV 341
#define FDIV 342
#define REM 343
#define UREM 344
#define SREM 345
#define FREM 346
#define AND 347
#define OR 348
#define XOR 349
#define SETLE 350
#define SETGE 351
#define SETLT 352
#define SETGT 353
#define SETEQ 354
#define SETNE 355
#define ICMP 356
#define FCMP 357
#define EQ 358
#define NE 359
#define SLT 360
#define SGT 361
#define SLE 362
#define SGE 363
#define OEQ 364
#define ONE 365
#define OLT 366
#define OGT 367
#define OLE 368
#define OGE 369
#define ORD 370
#define UNO 371
#define UEQ 372
#define UNE 373
#define ULT 374
#define UGT 375
#define ULE 376
#define UGE 377
#define MALLOC 378
#define ALLOCA 379
#define FREE 380
#define LOAD 381
#define STORE 382
#define GETELEMENTPTR 383
#define PHI_TOK 384
#define SELECT 385
#define SHL 386
#define SHR 387
#define ASHR 388
#define LSHR 389
#define VAARG 390
#define EXTRACTELEMENT 391
#define INSERTELEMENT 392
#define SHUFFLEVECTOR 393
#define CAST 394
#define TRUNC 395
#define ZEXT 396
#define SEXT 397
#define FPTRUNC 398
#define FPEXT 399
#define FPTOUI 400
#define FPTOSI 401
#define UITOFP 402
#define SITOFP 403
#define PTRTOINT 404
#define INTTOPTR 405
#define BITCAST 406




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 275 "/proj/llvm/llvm-1/tools/llvm-upgrade/UpgradeParser.y"
typedef union YYSTYPE {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
} YYSTYPE;
/* Line 1447 of yacc.c.  */
#line 348 "UpgradeParser.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE Upgradelval;



