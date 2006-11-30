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

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse Upgradeparse
#define yylex   Upgradelex
#define yyerror Upgradeerror
#define yylval  Upgradelval
#define yychar  Upgradechar
#define yydebug Upgradedebug
#define yynerrs Upgradenerrs


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




/* Copy the first part of user declarations.  */
#line 14 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"

#define YYERROR_VERBOSE 1
#define YYSTYPE std::string*

#include "ParserInternals.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/System/MappedFile.h>
#include <algorithm>
#include <list>
#include <utility>
#include <iostream>


int yylex();                       // declaration" of xxx warnings.
int yyparse();

static std::string CurFilename;

static std::ostream *O = 0;

void UpgradeAssembly(const std::string &infile, std::ostream &out)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  llvm::sys::Path p(infile);
  llvm::sys::MappedFile mf;
  mf.open(p);
  mf.map();
  const char* base = mf.charBase();
  size_t sz = mf.size();

  set_scan_bytes(base, sz);

  O = &out;

  if (yyparse()) {
    std::cerr << "Parse failed.\n";
    exit(1);
  }
}



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 393 "UpgradeParser.tab.c"

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T) && (defined (__STDC__) || defined (__cplusplus))
# include <stddef.h> /* INFRINGES ON USER NAME SPACE */
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

#if ! defined (yyoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if defined (__STDC__) || defined (__cplusplus)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     define YYINCLUDED_STDLIB_H
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2005 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM ((YYSIZE_T) -1)
#  endif
#  ifdef __cplusplus
extern "C" {
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if (! defined (malloc) && ! defined (YYINCLUDED_STDLIB_H) \
	&& (defined (__STDC__) || defined (__cplusplus)))
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if (! defined (free) && ! defined (YYINCLUDED_STDLIB_H) \
	&& (defined (__STDC__) || defined (__cplusplus)))
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifdef __cplusplus
}
#  endif
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (defined (YYSTYPE_IS_TRIVIAL) && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short int yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short int) + sizeof (YYSTYPE))			\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined (__GNUC__) && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (0)
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char yysigned_char;
#else
   typedef short int yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  4
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1246

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  138
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  71
/* YYNRULES -- Number of rules. */
#define YYNRULES  251
/* YYNRULES -- Number of states. */
#define YYNSTATES  510

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   378

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     127,   128,   136,     2,   125,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     132,   124,   133,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   129,   126,   131,     2,     2,     2,     2,     2,   137,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     130,     2,     2,   134,     2,   135,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short int yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    11,    13,    15,    17,
      19,    21,    23,    25,    27,    29,    31,    33,    35,    37,
      39,    41,    43,    45,    47,    49,    51,    53,    55,    57,
      59,    61,    63,    65,    67,    69,    71,    73,    75,    77,
      79,    82,    83,    85,    87,    89,    91,    93,    95,    97,
      98,   100,   102,   104,   106,   108,   110,   113,   114,   115,
     118,   119,   123,   126,   127,   129,   130,   134,   136,   139,
     141,   143,   145,   147,   149,   151,   153,   155,   157,   159,
     161,   163,   165,   167,   169,   171,   173,   175,   177,   179,
     181,   184,   189,   195,   201,   205,   208,   211,   213,   217,
     219,   223,   225,   226,   231,   235,   239,   244,   249,   253,
     256,   259,   262,   265,   268,   271,   274,   277,   280,   283,
     290,   296,   305,   312,   319,   326,   333,   340,   349,   358,
     362,   364,   366,   368,   370,   373,   376,   381,   384,   386,
     391,   394,   399,   406,   413,   420,   427,   431,   436,   437,
     439,   441,   443,   447,   451,   455,   459,   463,   467,   469,
     470,   472,   474,   476,   477,   480,   484,   486,   488,   492,
     494,   495,   504,   506,   508,   512,   514,   516,   520,   521,
     523,   525,   529,   530,   532,   534,   536,   538,   540,   542,
     544,   546,   548,   552,   554,   560,   562,   564,   566,   568,
     571,   574,   576,   580,   583,   584,   586,   589,   592,   596,
     606,   616,   625,   639,   641,   643,   650,   656,   659,   666,
     674,   676,   680,   682,   683,   686,   688,   694,   700,   706,
     709,   714,   719,   726,   731,   736,   743,   750,   753,   761,
     763,   766,   767,   769,   770,   774,   781,   785,   792,   795,
     800,   807
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     170,     0,    -1,     5,    -1,     6,    -1,     3,    -1,     4,
      -1,    78,    -1,    79,    -1,    80,    -1,    81,    -1,    82,
      -1,    83,    -1,    84,    -1,    85,    -1,    86,    -1,    87,
      -1,    88,    -1,    89,    -1,    90,    -1,    91,    -1,    92,
      -1,    93,    -1,    94,    -1,    95,    -1,   123,    -1,   116,
      -1,   117,    -1,   118,    -1,    16,    -1,    14,    -1,    12,
      -1,    10,    -1,    17,    -1,    15,    -1,    13,    -1,    11,
      -1,   146,    -1,   147,    -1,    18,    -1,    19,    -1,   178,
     124,    -1,    -1,    41,    -1,    42,    -1,    43,    -1,    44,
      -1,    45,    -1,    46,    -1,    47,    -1,    -1,    65,    -1,
      66,    -1,    67,    -1,    68,    -1,    69,    -1,    70,    -1,
      64,     4,    -1,    -1,    -1,    57,     4,    -1,    -1,   125,
      57,     4,    -1,    34,    24,    -1,    -1,   155,    -1,    -1,
     125,   158,   157,    -1,   155,    -1,    57,     4,    -1,   161,
      -1,     8,    -1,   163,    -1,     8,    -1,   163,    -1,     9,
      -1,    10,    -1,    11,    -1,    12,    -1,    13,    -1,    14,
      -1,    15,    -1,    16,    -1,    17,    -1,    18,    -1,    19,
      -1,    20,    -1,    21,    -1,    48,    -1,   162,    -1,   192,
      -1,   126,     4,    -1,   160,   127,   165,   128,    -1,   129,
       4,   130,   163,   131,    -1,   132,     4,   130,   163,   133,
      -1,   134,   164,   135,    -1,   134,   135,    -1,   163,   136,
      -1,   163,    -1,   164,   125,   163,    -1,   164,    -1,   164,
     125,    37,    -1,    37,    -1,    -1,   161,   129,   168,   131,
      -1,   161,   129,   131,    -1,   161,   137,    24,    -1,   161,
     132,   168,   133,    -1,   161,   134,   168,   135,    -1,   161,
     134,   135,    -1,   161,    38,    -1,   161,    39,    -1,   161,
     192,    -1,   161,   167,    -1,   161,    26,    -1,   146,   140,
      -1,   147,     4,    -1,     9,    27,    -1,     9,    28,    -1,
     149,     7,    -1,   144,   127,   166,    36,   161,   128,    -1,
     101,   127,   166,   206,   128,    -1,   115,   127,   166,   125,
     166,   125,   166,   128,    -1,   141,   127,   166,   125,   166,
     128,    -1,   142,   127,   166,   125,   166,   128,    -1,   143,
     127,   166,   125,   166,   128,    -1,   145,   127,   166,   125,
     166,   128,    -1,   120,   127,   166,   125,   166,   128,    -1,
     121,   127,   166,   125,   166,   125,   166,   128,    -1,   122,
     127,   166,   125,   166,   125,   166,   128,    -1,   168,   125,
     166,    -1,   166,    -1,    32,    -1,    33,    -1,   171,    -1,
     171,   187,    -1,   171,   189,    -1,   171,    62,    61,   173,
      -1,   171,    25,    -1,   172,    -1,   172,   150,    20,   159,
      -1,   172,   189,    -1,   172,    62,    61,   173,    -1,   172,
     150,   151,   169,   166,   157,    -1,   172,   150,    50,   169,
     161,   157,    -1,   172,   150,    45,   169,   161,   157,    -1,
     172,   150,    47,   169,   161,   157,    -1,   172,    51,   175,
      -1,   172,    58,   124,   176,    -1,    -1,    24,    -1,    56,
      -1,    55,    -1,    53,   124,   174,    -1,    54,   124,     4,
      -1,    52,   124,    24,    -1,    71,   124,    24,    -1,   129,
     177,   131,    -1,   177,   125,    24,    -1,    24,    -1,    -1,
      22,    -1,    24,    -1,   178,    -1,    -1,   161,   179,    -1,
     181,   125,   180,    -1,   180,    -1,   181,    -1,   181,   125,
      37,    -1,    37,    -1,    -1,   152,   159,   178,   127,   182,
     128,   156,   153,    -1,    29,    -1,   134,    -1,   151,   183,
     184,    -1,    30,    -1,   135,    -1,   185,   195,   186,    -1,
      -1,    45,    -1,    47,    -1,    31,   188,   183,    -1,    -1,
      63,    -1,     3,    -1,     4,    -1,     7,    -1,    27,    -1,
      28,    -1,    38,    -1,    39,    -1,    26,    -1,   132,   168,
     133,    -1,   167,    -1,    61,   190,    24,   125,    24,    -1,
     139,    -1,   178,    -1,   192,    -1,   191,    -1,   161,   193,
      -1,   195,   196,    -1,   196,    -1,   197,   150,   198,    -1,
     197,   200,    -1,    -1,    23,    -1,    72,   194,    -1,    72,
       8,    -1,    73,    21,   193,    -1,    73,     9,   193,   125,
      21,   193,   125,    21,   193,    -1,    74,   148,   193,   125,
      21,   193,   129,   199,   131,    -1,    74,   148,   193,   125,
      21,   193,   129,   131,    -1,    75,   152,   159,   193,   127,
     203,   128,    36,    21,   193,    76,    21,   193,    -1,    76,
      -1,    77,    -1,   199,   148,   191,   125,    21,   193,    -1,
     148,   191,   125,    21,   193,    -1,   150,   205,    -1,   161,
     129,   193,   125,   193,   131,    -1,   201,   125,   129,   193,
     125,   193,   131,    -1,   194,    -1,   202,   125,   194,    -1,
     202,    -1,    -1,    60,    59,    -1,    59,    -1,   141,   161,
     193,   125,   193,    -1,   142,   161,   193,   125,   193,    -1,
     143,   161,   193,   125,   193,    -1,    49,   194,    -1,   145,
     194,   125,   194,    -1,   144,   194,    36,   161,    -1,   115,
     194,   125,   194,   125,   194,    -1,   119,   194,   125,   161,
      -1,   120,   194,   125,   194,    -1,   121,   194,   125,   194,
     125,   194,    -1,   122,   194,   125,   194,   125,   194,    -1,
     114,   201,    -1,   204,   152,   159,   193,   127,   203,   128,
      -1,   208,    -1,   125,   202,    -1,    -1,    35,    -1,    -1,
      96,   161,   154,    -1,    96,   161,   125,    15,   193,   154,
      -1,    97,   161,   154,    -1,    97,   161,   125,    15,   193,
     154,    -1,    98,   194,    -1,   207,    99,   161,   193,    -1,
     207,   100,   194,   125,   161,   193,    -1,   101,   161,   193,
     206,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,    89,    89,    89,    90,    90,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    95,    95,    95,    96,    96,
      96,    96,    96,    96,    97,    98,    98,    98,   102,   102,
     102,   102,   103,   103,   103,   103,   104,   104,   105,   105,
     108,   112,   117,   117,   117,   117,   117,   117,   118,   119,
     122,   122,   122,   122,   122,   123,   123,   124,   129,   130,
     133,   134,   142,   148,   149,   152,   153,   162,   163,   176,
     176,   177,   177,   178,   182,   182,   182,   182,   182,   182,
     182,   183,   183,   183,   183,   183,   183,   184,   184,   184,
     188,   192,   197,   203,   209,   214,   217,   225,   225,   232,
     233,   238,   241,   251,   256,   259,   264,   269,   274,   277,
     282,   287,   292,   297,   303,   308,   313,   318,   323,   330,
     335,   337,   339,   341,   343,   345,   347,   349,   351,   356,
     358,   363,   363,   373,   378,   381,   386,   389,   392,   396,
     401,   406,   411,   416,   421,   426,   431,   436,   441,   446,
     448,   448,   451,   456,   461,   466,   473,   480,   485,   486,
     494,   494,   495,   495,   497,   503,   506,   510,   513,   517,
     520,   524,   539,   542,   546,   555,   556,   558,   564,   565,
     566,   570,   580,   582,   585,   585,   585,   585,   585,   586,
     586,   586,   587,   592,   593,   602,   602,   605,   605,   611,
     617,   619,   626,   630,   635,   638,   644,   649,   654,   659,
     665,   671,   677,   686,   691,   697,   702,   709,   716,   721,
     729,   730,   738,   739,   743,   748,   751,   756,   761,   766,
     771,   776,   781,   786,   791,   796,   801,   806,   811,   820,
     825,   829,   833,   834,   837,   844,   851,   858,   865,   870,
     877,   884
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ESINT64VAL", "EUINT64VAL", "SINTVAL",
  "UINTVAL", "FPVAL", "VOID", "BOOL", "SBYTE", "UBYTE", "SHORT", "USHORT",
  "INT", "UINT", "LONG", "ULONG", "FLOAT", "DOUBLE", "TYPE", "LABEL",
  "VAR_ID", "LABELSTR", "STRINGCONSTANT", "IMPLEMENTATION",
  "ZEROINITIALIZER", "TRUETOK", "FALSETOK", "BEGINTOK", "ENDTOK",
  "DECLARE", "GLOBAL", "CONSTANT", "SECTION", "VOLATILE", "TO",
  "DOTDOTDOT", "NULL_TOK", "UNDEF", "CONST", "INTERNAL", "LINKONCE",
  "WEAK", "APPENDING", "DLLIMPORT", "DLLEXPORT", "EXTERN_WEAK", "OPAQUE",
  "NOT", "EXTERNAL", "TARGET", "TRIPLE", "ENDIAN", "POINTERSIZE", "LITTLE",
  "BIG", "ALIGN", "DEPLIBS", "CALL", "TAIL", "ASM_TOK", "MODULE",
  "SIDEEFFECT", "CC_TOK", "CCC_TOK", "CSRETCC_TOK", "FASTCC_TOK",
  "COLDCC_TOK", "X86_STDCALLCC_TOK", "X86_FASTCALLCC_TOK", "DATALAYOUT",
  "RET", "BR", "SWITCH", "INVOKE", "UNWIND", "UNREACHABLE", "ADD", "SUB",
  "MUL", "UDIV", "SDIV", "FDIV", "UREM", "SREM", "FREM", "AND", "OR",
  "XOR", "SETLE", "SETGE", "SETLT", "SETGT", "SETEQ", "SETNE", "MALLOC",
  "ALLOCA", "FREE", "LOAD", "STORE", "GETELEMENTPTR", "TRUNC", "ZEXT",
  "SEXT", "FPTRUNC", "FPEXT", "BITCAST", "UITOFP", "SITOFP", "FPTOUI",
  "FPTOSI", "INTTOPTR", "PTRTOINT", "PHI_TOK", "SELECT", "SHL", "LSHR",
  "ASHR", "VAARG", "EXTRACTELEMENT", "INSERTELEMENT", "SHUFFLEVECTOR",
  "CAST", "'='", "','", "'\\\\'", "'('", "')'", "'['", "'x'", "']'", "'<'",
  "'>'", "'{'", "'}'", "'*'", "'c'", "$accept", "INTVAL", "EINT64VAL",
  "ArithmeticOps", "LogicalOps", "SetCondOps", "CastOps", "ShiftOps",
  "SIntType", "UIntType", "IntType", "FPType", "OptAssign", "OptLinkage",
  "OptCallingConv", "OptAlign", "OptCAlign", "SectionString", "OptSection",
  "GlobalVarAttributes", "GlobalVarAttribute", "TypesV", "UpRTypesV",
  "Types", "PrimType", "UpRTypes", "TypeListI", "ArgTypeListI", "ConstVal",
  "ConstExpr", "ConstVector", "GlobalType", "Module", "DefinitionList",
  "ConstPool", "AsmBlock", "BigOrLittle", "TargetDefinition",
  "LibrariesDefinition", "LibList", "Name", "OptName", "ArgVal",
  "ArgListH", "ArgList", "FunctionHeaderH", "BEGIN", "FunctionHeader",
  "END", "Function", "FnDeclareLinkage", "FunctionProto", "OptSideEffect",
  "ConstValueRef", "SymbolicValueRef", "ValueRef", "ResolvedVal",
  "BasicBlockList", "BasicBlock", "InstructionList", "BBTerminatorInst",
  "JumpTable", "Inst", "PHIList", "ValueRefList", "ValueRefListE",
  "OptTailCall", "InstVal", "IndexList", "OptVolatile", "MemoryInst", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short int yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,    61,    44,    92,    40,    41,    91,
     120,    93,    60,    62,   123,   125,    42,    99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   138,   139,   139,   140,   140,   141,   141,   141,   141,
     141,   141,   141,   141,   141,   142,   142,   142,   143,   143,
     143,   143,   143,   143,   144,   145,   145,   145,   146,   146,
     146,   146,   147,   147,   147,   147,   148,   148,   149,   149,
     150,   150,   151,   151,   151,   151,   151,   151,   151,   151,
     152,   152,   152,   152,   152,   152,   152,   152,   153,   153,
     154,   154,   155,   156,   156,   157,   157,   158,   158,   159,
     159,   160,   160,   161,   162,   162,   162,   162,   162,   162,
     162,   162,   162,   162,   162,   162,   162,   163,   163,   163,
     163,   163,   163,   163,   163,   163,   163,   164,   164,   165,
     165,   165,   165,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   166,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   168,
     168,   169,   169,   170,   171,   171,   171,   171,   171,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   173,
     174,   174,   175,   175,   175,   175,   176,   177,   177,   177,
     178,   178,   179,   179,   180,   181,   181,   182,   182,   182,
     182,   183,   184,   184,   185,   186,   186,   187,   188,   188,
     188,   189,   190,   190,   191,   191,   191,   191,   191,   191,
     191,   191,   191,   191,   191,   192,   192,   193,   193,   194,
     195,   195,   196,   197,   197,   197,   198,   198,   198,   198,
     198,   198,   198,   198,   198,   199,   199,   200,   201,   201,
     202,   202,   203,   203,   204,   204,   205,   205,   205,   205,
     205,   205,   205,   205,   205,   205,   205,   205,   205,   205,
     206,   206,   207,   207,   208,   208,   208,   208,   208,   208,
     208,   208
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     0,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     1,     1,     1,     1,     1,     2,     0,     0,     2,
       0,     3,     2,     0,     1,     0,     3,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     4,     5,     5,     3,     2,     2,     1,     3,     1,
       3,     1,     0,     4,     3,     3,     4,     4,     3,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     6,
       5,     8,     6,     6,     6,     6,     6,     8,     8,     3,
       1,     1,     1,     1,     2,     2,     4,     2,     1,     4,
       2,     4,     6,     6,     6,     6,     3,     4,     0,     1,
       1,     1,     3,     3,     3,     3,     3,     3,     1,     0,
       1,     1,     1,     0,     2,     3,     1,     1,     3,     1,
       0,     8,     1,     1,     3,     1,     1,     3,     0,     1,
       1,     3,     0,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     1,     5,     1,     1,     1,     1,     2,
       2,     1,     3,     2,     0,     1,     2,     2,     3,     9,
       9,     8,    13,     1,     1,     6,     5,     2,     6,     7,
       1,     3,     1,     0,     2,     1,     5,     5,     5,     2,
       4,     4,     6,     4,     4,     6,     6,     2,     7,     1,
       2,     0,     1,     0,     3,     6,     3,     6,     2,     4,
       6,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
     148,     0,    49,   138,     1,   137,   178,    42,    43,    44,
      45,    46,    47,    48,     0,    57,   204,   134,   135,   160,
     161,     0,     0,     0,    49,     0,   140,   179,   180,    57,
       0,     0,    50,    51,    52,    53,    54,    55,     0,     0,
     205,   204,   201,    41,     0,     0,     0,     0,   146,     0,
       0,     0,     0,     0,     0,     0,    40,   181,   149,   136,
      56,     2,     3,    70,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,     0,     0,
       0,     0,   195,     0,     0,    69,    88,    73,   196,    89,
     172,   173,   174,   175,   176,   177,   200,   243,   203,     0,
       0,     0,     0,   159,   147,   141,   139,   131,   132,     0,
       0,     0,     0,    90,     0,     0,    72,    95,    97,     0,
       0,   102,    96,   242,     0,   225,     0,     0,     0,     0,
      57,   213,   214,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,     0,     0,     0,     0,     0,     0,    25,    26,    27,
       0,     0,     0,     0,    24,     0,     0,     0,     0,     0,
     202,    57,   217,     0,   239,   154,   151,   150,   152,   153,
     155,   158,     0,    65,    65,    65,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,     0,     0,     0,
       0,    65,     0,     0,     0,    94,   170,   101,    99,     0,
       0,   229,   224,   207,   206,     0,     0,    31,    35,    30,
      34,    29,    33,    28,    32,    36,    37,     0,     0,    60,
      60,   248,     0,     0,   237,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   156,
       0,   144,   145,   143,   116,   117,     4,     5,   114,   115,
     118,   113,   109,   110,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   112,   111,
     142,    71,    71,    98,   169,   163,   166,   167,     0,     0,
      91,   184,   185,   186,   191,   187,   188,   189,   190,   182,
       0,   193,   198,   197,   199,     0,   208,     0,     0,     0,
     244,     0,   246,   241,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   157,
       0,     0,    67,    65,     0,     0,     0,     0,     0,   104,
     130,     0,     0,   108,     0,   105,     0,     0,     0,     0,
       0,    92,    93,   162,   164,     0,    63,   100,   183,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   251,     0,
       0,     0,   233,   234,     0,     0,     0,     0,     0,   231,
     230,     0,   249,     0,    62,    68,    66,   241,     0,     0,
       0,     0,     0,   103,   106,   107,     0,     0,     0,     0,
       0,   168,   165,    64,    58,     0,   192,     0,     0,   223,
      60,    61,    60,   220,   240,     0,     0,     0,     0,     0,
     226,   227,   228,   223,     0,     0,     0,     0,     0,     0,
     129,     0,     0,     0,     0,     0,     0,   171,     0,     0,
       0,   222,     0,     0,   245,   247,     0,     0,     0,   232,
     235,   236,     0,   250,   120,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    59,   194,     0,     0,     0,   221,
     218,     0,   238,     0,   126,     0,     0,   122,   123,   124,
     119,   125,     0,   211,     0,     0,     0,   219,     0,     0,
       0,   209,     0,   210,     0,     0,   121,   127,   128,     0,
       0,     0,     0,     0,     0,   216,     0,     0,   215,   212
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,    82,   258,   273,   274,   275,   276,   277,   197,   198,
     227,   199,    24,    15,    38,   437,   310,   332,   404,   251,
     333,    83,    84,   200,    86,    87,   119,   209,   340,   301,
     341,   109,     1,     2,     3,    59,   178,    48,   104,   182,
      88,   354,   286,   287,   288,    39,    92,    16,    95,    17,
      29,    18,   359,   302,    89,   304,   413,    41,    42,    43,
     170,   485,    98,   234,   441,   442,   171,   172,   368,   173,
     174
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -451
static const short int yypact[] =
{
    -451,    18,   670,   213,  -451,  -451,    63,  -451,  -451,  -451,
    -451,  -451,  -451,  -451,   -39,   308,    33,  -451,  -451,  -451,
    -451,    14,   -36,    31,   150,   -12,  -451,  -451,  -451,   308,
      96,   136,  -451,  -451,  -451,  -451,  -451,  -451,   971,   -13,
    -451,    -9,  -451,   119,    10,    20,    37,    56,  -451,    40,
      96,   971,    45,    45,    45,    45,  -451,  -451,  -451,  -451,
    -451,  -451,  -451,    52,  -451,  -451,  -451,  -451,  -451,  -451,
    -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,   177,   179,
     180,    85,  -451,   119,    58,  -451,  -451,   -52,  -451,  -451,
    -451,  -451,  -451,  -451,  -451,  -451,  -451,  1123,  -451,   163,
      97,   185,   174,   178,  -451,  -451,  -451,  -451,  -451,  1019,
    1019,  1019,  1060,  -451,    76,    82,  -451,  -451,   -52,   -85,
      86,   786,  -451,  -451,  1019,  -451,   157,  1101,    30,   278,
     308,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,
    -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,
    -451,  1019,  1019,  1019,  1019,  1019,  1019,  -451,  -451,  -451,
    1019,  1019,  1019,  1019,  -451,  1019,  1019,  1019,  1019,  1019,
    -451,   308,  -451,    55,  -451,  -451,  -451,  -451,  -451,  -451,
    -451,  -451,   -49,    90,    90,    90,   130,   156,   214,   160,
     217,   162,   218,   168,   220,   224,   225,   170,   221,   227,
     387,    90,  1019,  1019,  1019,  -451,   827,  -451,   111,   122,
     560,  -451,  -451,    52,  -451,   560,   560,  -451,  -451,  -451,
    -451,  -451,  -451,  -451,  -451,  -451,  -451,   560,   971,   123,
     126,  -451,   560,   118,   127,   128,   140,   141,   142,   143,
     560,   560,   560,   233,   145,   971,  1019,  1019,   248,  -451,
     -23,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,
    -451,  -451,  -451,  -451,   146,   147,   149,   169,   171,   868,
    1060,   537,   253,   173,   181,   182,   183,   187,  -451,  -451,
    -451,   -94,     6,   -52,  -451,   119,  -451,   158,   188,   889,
    -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,  -451,   234,
    1060,  -451,  -451,  -451,  -451,   190,  -451,   197,   560,    -5,
    -451,     8,  -451,   201,   560,   199,  1019,  1019,  1019,  1019,
    1019,   205,   207,   208,  1019,  1019,   560,   560,   209,  -451,
     282,   303,  -451,    90,  1060,  1060,  1060,  1060,  1060,  -451,
    -451,   -44,   -89,  -451,   -56,  -451,  1060,  1060,  1060,  1060,
    1060,  -451,  -451,  -451,  -451,   930,   304,  -451,  -451,   313,
     -14,   320,   323,   219,   560,   344,   560,  1019,  -451,   226,
     560,   228,  -451,  -451,   229,   230,   560,   560,   560,  -451,
    -451,   222,  -451,  1019,  -451,  -451,  -451,   201,   231,   238,
     239,   241,  1060,  -451,  -451,  -451,   244,   245,   257,   348,
     261,  -451,  -451,  -451,   330,   263,  -451,   560,   560,  1019,
     264,  -451,   264,  -451,   265,   560,   266,  1019,  1019,  1019,
    -451,  -451,  -451,  1019,   560,   267,  1060,  1060,  1060,  1060,
    -451,  1060,  1060,  1060,  1019,  1060,   390,  -451,   373,   273,
     270,   265,   272,   345,  -451,  -451,  1019,   274,   560,  -451,
    -451,  -451,   275,  -451,  -451,   276,   279,   281,   285,   284,
     286,   287,   288,   290,  -451,  -451,   383,    15,   384,  -451,
    -451,   291,  -451,  1060,  -451,  1060,  1060,  -451,  -451,  -451,
    -451,  -451,   560,  -451,   661,    47,   398,  -451,   293,   295,
     296,  -451,   302,  -451,   661,   560,  -451,  -451,  -451,   408,
     306,   356,   560,   412,   414,  -451,   560,   560,  -451,  -451
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -451,  -451,  -451,   339,   341,   342,   343,   346,  -128,  -127,
    -450,  -451,   401,   421,   -97,  -451,  -224,    92,  -451,  -177,
    -451,   -46,  -451,   -38,  -451,   -66,   328,  -451,  -108,   250,
    -251,    94,  -451,  -451,  -451,   402,  -451,  -451,  -451,  -451,
       0,  -451,    98,  -451,  -451,   422,  -451,  -451,  -451,  -451,
    -451,   451,  -451,  -414,   -65,   120,  -115,  -451,   415,  -451,
    -451,  -451,  -451,  -451,    88,    34,  -451,  -451,    71,  -451,
    -451
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -134
static const short int yytable[] =
{
      85,   225,   226,    25,   201,   106,   312,   252,   253,   211,
     364,   330,   214,    85,    40,   118,    90,   484,     4,   342,
     344,    93,    30,   366,   280,   217,   218,   219,   220,   221,
     222,   223,   224,   228,   331,   494,   392,   351,   231,   215,
     204,   235,   122,    25,   394,   236,   237,   238,   239,   360,
     205,   216,   365,   243,   244,   118,    40,   217,   218,   219,
     220,   221,   222,   223,   224,   365,    44,    45,    46,   392,
     492,   183,   184,   185,   245,   -71,   248,   107,   108,   395,
     500,   392,   249,   120,   122,    47,   210,   393,    49,   210,
      61,    62,    50,   116,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    19,    27,    20,
      28,   392,    56,   229,   230,   210,   232,   233,   210,   406,
      58,    91,   210,   210,   210,   210,    94,   240,   241,   242,
     210,   210,   328,    77,    99,   279,   281,   282,   283,   352,
      60,    19,   122,    20,   100,   303,   483,   110,   111,   112,
     303,   303,   176,   177,   246,   247,   386,   254,   255,   -31,
     -31,   101,   303,   -30,   -30,   -29,   -29,   303,   285,   103,
      51,   -28,   -28,   256,   257,   303,   303,   303,   493,   -72,
     102,   113,   308,   114,   115,   121,   444,   175,   445,   179,
      85,     7,     8,     9,    10,    52,    12,    53,   180,   326,
      54,   371,   181,   373,   374,   375,   202,    85,   327,   210,
     380,    78,   203,   206,    79,   250,   212,    80,   -35,    81,
     117,   -34,   -33,   283,   -32,   259,   387,   388,   389,   390,
     391,   -38,   -39,   -41,   260,    19,   289,    20,   396,   397,
     398,   399,   400,   303,     6,   -41,   -41,   314,   309,   303,
     290,   311,   315,   316,   -41,   -41,   -41,   -41,   -41,   -41,
     -41,   303,   303,   -41,    21,   317,   318,   319,   320,   324,
     325,    22,   329,   334,   335,    23,   336,   345,   210,   372,
     210,   210,   210,   355,   430,   353,   379,   210,   217,   218,
     219,   220,   221,   222,   223,   224,   337,   358,   338,   303,
     346,   303,   449,   450,   451,   303,   384,   385,   347,   348,
     349,   303,   303,   303,   350,   361,   356,   285,   455,   456,
     457,   458,   362,   459,   460,   461,   367,   463,   370,   210,
     376,   469,   377,   378,   383,   305,   306,   405,   330,   225,
     226,   407,   303,   303,   408,   424,   409,   307,   411,   423,
     303,   415,   313,   417,   418,   419,   426,   225,   226,   303,
     321,   322,   323,   427,   428,   488,   429,   489,   490,   431,
     432,   210,    31,    32,    33,    34,    35,    36,    37,   210,
     210,   210,   433,   303,   434,   210,   435,   436,   438,   443,
     446,   448,    61,    62,   464,   454,   462,   465,   466,   467,
     468,   473,   365,   472,   482,   470,   475,   474,   210,    19,
     476,    20,   477,   261,   478,   479,   480,   303,   481,   495,
     486,   496,   487,   497,   498,   262,   263,   499,   363,   502,
     303,   503,   504,   506,   369,   507,   165,   303,   166,   167,
     168,   303,   303,   169,    97,    55,   381,   382,   403,   208,
     278,    57,   105,   402,    26,   414,    96,   452,   425,     0,
       0,     0,     0,     0,     0,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,     0,   410,     0,   412,     0,   264,     0,
     416,     0,     0,     0,     0,     0,   420,   421,   422,     0,
       0,     0,   265,   157,   158,   159,     0,   266,   267,   268,
     164,     0,     0,     0,     0,     0,   269,     0,     0,   270,
       0,   271,     0,     0,   272,     0,     0,   439,   440,     0,
       0,     0,     0,     0,     0,   447,     0,     0,     0,     0,
       0,     0,    61,    62,   453,   116,   186,   187,   188,   189,
     190,   191,   192,   193,   194,   195,   196,    75,    76,    19,
       0,    20,     0,   291,   292,    61,    62,   293,   471,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    19,     0,    20,    77,   294,   295,   296,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   297,   298,
       0,     0,   491,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   501,     0,     0,     0,     0,
       0,   299,   505,     0,     0,     0,   508,   509,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,     0,     0,     0,     0,
       0,   264,     0,    78,   291,   292,    79,     0,   293,    80,
    -133,    81,   343,     0,     0,   265,   157,   158,   159,     0,
     266,   267,   268,   164,     0,     0,     0,   294,   295,   296,
       0,     0,   300,     0,     0,     5,     0,     0,     0,   297,
     298,     6,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     7,     8,     9,    10,    11,    12,    13,     0,     0,
       0,     0,   299,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    14,     0,     0,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,     0,     0,     0,
       0,     0,   264,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   265,   157,   158,   159,
       0,   266,   267,   268,   164,     0,     0,     0,     0,     0,
       0,    61,    62,   300,   116,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    19,     0,
      20,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   207,     0,     0,     0,     0,     0,     0,
       0,     0,    61,    62,    77,   116,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    19,
       0,    20,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   284,     0,     0,     0,     0,     0,
       0,     0,     0,    61,    62,    77,   116,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,    75,    76,
      19,     0,    20,     0,    61,    62,     0,   116,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    19,    78,    20,     0,    79,    77,     0,    80,     0,
      81,     0,     0,     0,     0,     0,   357,     0,     0,     0,
       0,     0,     0,     0,     0,    61,    62,    77,   116,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    19,    78,    20,     0,    79,     0,     0,    80,
       0,    81,     0,     0,     0,     0,     0,   401,     0,     0,
       0,     0,     0,     0,     0,     0,    61,    62,    77,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    19,    78,    20,     0,    79,     0,   339,
      80,     0,    81,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    78,     0,     0,    79,    77,
       0,    80,     0,    81,    61,    62,     0,   116,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    19,     0,    20,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    78,     0,     0,    79,
       0,     0,    80,     0,    81,    61,    62,    77,   116,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
      75,    76,    19,     0,    20,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    78,     0,     0,
      79,     0,     0,    80,     0,    81,    61,    62,    77,   213,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    19,     0,    20,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    78,     0,     0,    79,    77,
       0,    80,     0,    81,     0,     0,     0,     0,   123,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   124,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   125,   126,     0,     0,    78,     0,     0,    79,
       0,     0,    80,     0,    81,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,     0,     0,   154,     0,     0,    78,     0,     0,
      79,     0,     0,    80,     0,    81,     0,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   164
};

static const short int yycheck[] =
{
      38,   129,   129,     3,   112,    51,   230,   184,   185,   124,
      15,    34,   127,    51,    23,    81,    29,   467,     0,   270,
     271,    30,    61,    15,   201,    10,    11,    12,    13,    14,
      15,    16,    17,   130,    57,   485,   125,   131,   153,     9,
     125,   156,   136,    43,   133,   160,   161,   162,   163,   300,
     135,    21,    57,   168,   169,   121,    23,    10,    11,    12,
      13,    14,    15,    16,    17,    57,    52,    53,    54,   125,
     484,   109,   110,   111,   171,   127,   125,    32,    33,   135,
     494,   125,   131,    83,   136,    71,   124,   131,   124,   127,
       5,     6,    61,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    45,    24,
      47,   125,   124,   151,   152,   153,   154,   155,   156,   133,
      24,   134,   160,   161,   162,   163,   135,   165,   166,   167,
     168,   169,   247,    48,   124,   200,   202,   203,   204,   133,
       4,    22,   136,    24,   124,   210,   131,    53,    54,    55,
     215,   216,    55,    56,    99,   100,   333,    27,    28,     3,
       4,   124,   227,     3,     4,     3,     4,   232,   206,   129,
      20,     3,     4,     3,     4,   240,   241,   242,   131,   127,
     124,     4,   228,     4,     4,   127,   410,    24,   412,     4,
     228,    41,    42,    43,    44,    45,    46,    47,    24,   245,
      50,   316,    24,   318,   319,   320,   130,   245,   246,   247,
     325,   126,   130,   127,   129,   125,    59,   132,     4,   134,
     135,     4,     4,   289,     4,     4,   334,   335,   336,   337,
     338,     7,     7,    20,     7,    22,   125,    24,   346,   347,
     348,   349,   350,   308,    31,    32,    33,   129,   125,   314,
     128,   125,   125,   125,    41,    42,    43,    44,    45,    46,
      47,   326,   327,    50,    51,   125,   125,   125,   125,    36,
     125,    58,    24,   127,   127,    62,   127,    24,   316,   317,
     318,   319,   320,   125,   392,   285,   324,   325,    10,    11,
      12,    13,    14,    15,    16,    17,   127,    63,   127,   364,
     127,   366,   417,   418,   419,   370,    24,     4,   127,   127,
     127,   376,   377,   378,   127,   125,   128,   355,   426,   427,
     428,   429,   125,   431,   432,   433,   125,   435,   129,   367,
     125,   446,   125,   125,   125,   215,   216,    24,    34,   467,
     467,    21,   407,   408,    21,   383,   127,   227,     4,   127,
     415,   125,   232,   125,   125,   125,   125,   485,   485,   424,
     240,   241,   242,   125,   125,   473,   125,   475,   476,   125,
     125,   409,    64,    65,    66,    67,    68,    69,    70,   417,
     418,   419,   125,   448,    36,   423,   125,    57,   125,   125,
     125,   125,     5,     6,     4,   128,   434,    24,   125,   129,
     128,   125,    57,   128,    21,   131,   125,   128,   446,    22,
     125,    24,   128,    26,   128,   128,   128,   482,   128,    21,
      36,   128,   131,   128,   128,    38,    39,   125,   308,    21,
     495,   125,    76,    21,   314,    21,    97,   502,    97,    97,
      97,   506,   507,    97,    43,    24,   326,   327,   356,   121,
     200,    29,    50,   355,     3,   367,    41,   423,   387,    -1,
      -1,    -1,    -1,    -1,    -1,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    -1,   364,    -1,   366,    -1,   101,    -1,
     370,    -1,    -1,    -1,    -1,    -1,   376,   377,   378,    -1,
      -1,    -1,   115,   116,   117,   118,    -1,   120,   121,   122,
     123,    -1,    -1,    -1,    -1,    -1,   129,    -1,    -1,   132,
      -1,   134,    -1,    -1,   137,    -1,    -1,   407,   408,    -1,
      -1,    -1,    -1,    -1,    -1,   415,    -1,    -1,    -1,    -1,
      -1,    -1,     5,     6,   424,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      -1,    24,    -1,     3,     4,     5,     6,     7,   448,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    22,    -1,    24,    48,    26,    27,    28,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    39,
      -1,    -1,   482,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   495,    -1,    -1,    -1,    -1,
      -1,    61,   502,    -1,    -1,    -1,   506,   507,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    95,    -1,    -1,    -1,    -1,
      -1,   101,    -1,   126,     3,     4,   129,    -1,     7,   132,
       0,   134,   135,    -1,    -1,   115,   116,   117,   118,    -1,
     120,   121,   122,   123,    -1,    -1,    -1,    26,    27,    28,
      -1,    -1,   132,    -1,    -1,    25,    -1,    -1,    -1,    38,
      39,    31,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    41,    42,    43,    44,    45,    46,    47,    -1,    -1,
      -1,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    62,    -1,    -1,    -1,    -1,    -1,    -1,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    -1,    -1,    -1,
      -1,    -1,   101,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,   116,   117,   118,
      -1,   120,   121,   122,   123,    -1,    -1,    -1,    -1,    -1,
      -1,     5,     6,   132,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    -1,
      24,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     5,     6,    48,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      -1,    24,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     5,     6,    48,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    -1,    24,    -1,     5,     6,    -1,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,   126,    24,    -1,   129,    48,    -1,   132,    -1,
     134,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     5,     6,    48,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,   126,    24,    -1,   129,    -1,    -1,   132,
      -1,   134,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     5,     6,    48,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,   126,    24,    -1,   129,    -1,   131,
     132,    -1,   134,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,   129,    48,
      -1,   132,    -1,   134,     5,     6,    -1,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    -1,    24,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,   129,
      -1,    -1,   132,    -1,   134,     5,     6,    48,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    -1,    24,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,
     129,    -1,    -1,   132,    -1,   134,     5,     6,    48,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    -1,    24,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   126,    -1,    -1,   129,    48,
      -1,   132,    -1,   134,    -1,    -1,    -1,    -1,    35,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    49,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    59,    60,    -1,    -1,   126,    -1,    -1,   129,
      -1,    -1,   132,    -1,   134,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    -1,    -1,   101,    -1,    -1,   126,    -1,    -1,
     129,    -1,    -1,   132,    -1,   134,    -1,   114,   115,   116,
     117,   118,   119,   120,   121,   122,   123
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   170,   171,   172,     0,    25,    31,    41,    42,    43,
      44,    45,    46,    47,    62,   151,   185,   187,   189,    22,
      24,    51,    58,    62,   150,   178,   189,    45,    47,   188,
      61,    64,    65,    66,    67,    68,    69,    70,   152,   183,
      23,   195,   196,   197,    52,    53,    54,    71,   175,   124,
      61,    20,    45,    47,    50,   151,   124,   183,    24,   173,
       4,     5,     6,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    48,   126,   129,
     132,   134,   139,   159,   160,   161,   162,   163,   178,   192,
      29,   134,   184,    30,   135,   186,   196,   150,   200,   124,
     124,   124,   124,   129,   176,   173,   159,    32,    33,   169,
     169,   169,   169,     4,     4,     4,     8,   135,   163,   164,
     178,   127,   136,    35,    49,    59,    60,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,   101,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   141,   142,   143,   144,   145,
     198,   204,   205,   207,   208,    24,    55,    56,   174,     4,
      24,    24,   177,   161,   161,   161,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,   146,   147,   149,
     161,   166,   130,   130,   125,   135,   127,    37,   164,   165,
     161,   194,    59,     8,   194,     9,    21,    10,    11,    12,
      13,    14,    15,    16,    17,   146,   147,   148,   152,   161,
     161,   194,   161,   161,   201,   194,   194,   194,   194,   194,
     161,   161,   161,   194,   194,   152,    99,   100,   125,   131,
     125,   157,   157,   157,    27,    28,     3,     4,   140,     4,
       7,    26,    38,    39,   101,   115,   120,   121,   122,   129,
     132,   134,   137,   141,   142,   143,   144,   145,   167,   192,
     157,   163,   163,   163,    37,   161,   180,   181,   182,   125,
     128,     3,     4,     7,    26,    27,    28,    38,    39,    61,
     132,   167,   191,   192,   193,   193,   193,   193,   159,   125,
     154,   125,   154,   193,   129,   125,   125,   125,   125,   125,
     125,   193,   193,   193,    36,   125,   159,   161,   194,    24,
      34,    57,   155,   158,   127,   127,   127,   127,   127,   131,
     166,   168,   168,   135,   168,    24,   127,   127,   127,   127,
     127,   131,   133,   178,   179,   125,   128,    37,    63,   190,
     168,   125,   125,   193,    15,    57,    15,   125,   206,   193,
     129,   194,   161,   194,   194,   194,   125,   125,   125,   161,
     194,   193,   193,   125,    24,     4,   157,   166,   166,   166,
     166,   166,   125,   131,   133,   135,   166,   166,   166,   166,
     166,    37,   180,   155,   156,    24,   133,    21,    21,   127,
     193,     4,   193,   194,   202,   125,   193,   125,   125,   125,
     193,   193,   193,   127,   161,   206,   125,   125,   125,   125,
     166,   125,   125,   125,    36,   125,    57,   153,   125,   193,
     193,   202,   203,   125,   154,   154,   125,   193,   125,   194,
     194,   194,   203,   193,   128,   166,   166,   166,   166,   166,
     166,   166,   161,   166,     4,    24,   125,   129,   128,   194,
     131,   193,   128,   125,   128,   125,   125,   128,   128,   128,
     128,   128,    21,   131,   148,   199,    36,   131,   166,   166,
     166,   193,   191,   131,   148,    21,   128,   128,   128,   125,
     191,   193,    21,   125,    76,   193,    21,    21,   193,   193
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (0)


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (N)								\
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (0)
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
              (Loc).first_line, (Loc).first_column,	\
              (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (0)

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr,					\
                  Type, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short int *bottom, short int *top)
#else
static void
yy_stack_print (bottom, top)
    short int *bottom;
    short int *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_reduce_print (int yyrule)
#else
static void
yy_reduce_print (yyrule)
    int yyrule;
#endif
{
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu), ",
             yyrule - 1, yylno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname[yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname[yyr1[yyrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (Rule);		\
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
yystrlen (const char *yystr)
#   else
yystrlen (yystr)
     const char *yystr;
#   endif
{
  const char *yys = yystr;

  while (*yys++ != '\0')
    continue;

  return yys - yystr - 1;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
yystpcpy (char *yydest, const char *yysrc)
#   else
yystpcpy (yydest, yysrc)
     char *yydest;
     const char *yysrc;
#   endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      size_t yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

#endif /* YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yysymprint (FILE *yyoutput, int yytype, YYSTYPE *yyvaluep)
#else
static void
yysymprint (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);


# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  switch (yytype)
    {
      default:
        break;
    }
  YYFPRINTF (yyoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
        break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM);
# else
int yyparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM)
# else
int yyparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short int yyssa[YYINITDEPTH];
  short int *yyss = yyssa;
  short int *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK   (yyvsp--, yyssp--)

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int yylen;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.
     */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	short int *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short int *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

/* Do appropriate processing given the current state.  */
/* Read a look-ahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to look-ahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 40:
#line 108 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1])->append(" = ");
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 41:
#line 112 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string(""); 
  ;}
    break;

  case 49:
#line 119 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(""); ;}
    break;

  case 57:
#line 124 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(""); ;}
    break;

  case 58:
#line 129 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 59:
#line 130 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { *(yyvsp[-1]) += " " + *(yyvsp[0]); delete (yyvsp[0]); (yyval) = (yyvsp[-1]); ;}
    break;

  case 60:
#line 133 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 61:
#line 134 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1])->insert(0, ", "); 
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 62:
#line 142 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 63:
#line 148 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 65:
#line 152 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 66:
#line 153 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      (yyvsp[-1])->insert(0, ", ");
      if (!(yyvsp[0])->empty())
        *(yyvsp[-1]) += " " + *(yyvsp[0]);
      delete (yyvsp[0]);
      (yyval) = (yyvsp[-1]);
    ;}
    break;

  case 68:
#line 163 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      *(yyvsp[-1]) += " " + *(yyvsp[0]);
      delete (yyvsp[0]);
      (yyval) = (yyvsp[-1]);
    ;}
    break;

  case 90:
#line 188 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Type UpReference
    (yyvsp[0])->insert(0, "\\");
    (yyval) = (yyvsp[0]);
  ;}
    break;

  case 91:
#line 192 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {           // Function derived type?
    *(yyvsp[-3]) += "( " + *(yyvsp[-1]) + " )";
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 92:
#line 197 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Sized array type?
    (yyvsp[-3])->insert(0,"[ ");
    *(yyvsp[-3]) += " x " + *(yyvsp[-1]) + " ]";
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 93:
#line 203 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Packed array type?
    (yyvsp[-3])->insert(0,"< ");
    *(yyvsp[-3]) += " x " + *(yyvsp[-1]) + " >";
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 94:
#line 209 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                        // Structure type?
    (yyvsp[-1])->insert(0, "{ ");
    *(yyvsp[-1]) += " }";
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 95:
#line 214 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                  // Empty structure type?
    (yyval) = new std::string("{ }");
  ;}
    break;

  case 96:
#line 217 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                             // Pointer type?
    *(yyvsp[-1]) += '*';
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 98:
#line 225 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += ", " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 100:
#line 233 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += ", ...";
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 101:
#line 238 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = (yyvsp[0]);
  ;}
    break;

  case 102:
#line 241 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string();
  ;}
    break;

  case 103:
#line 251 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    *(yyvsp[-3]) += " [ " + *(yyvsp[-1]) + " ]";
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 104:
#line 256 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string("[ ]");
  ;}
    break;

  case 105:
#line 259 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " c" + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 106:
#line 264 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    *(yyvsp[-3]) += " < " + *(yyvsp[-1]) + " >";
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 107:
#line 269 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3]) += " { " + *(yyvsp[-1]) + " }";
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 108:
#line 274 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string("[ ]");
  ;}
    break;

  case 109:
#line 277 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]); 
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 110:
#line 282 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 111:
#line 287 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 112:
#line 292 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 113:
#line 297 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 114:
#line 303 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {      // integral constants
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 115:
#line 308 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {            // integral constants
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 116:
#line 313 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                      // Boolean constants
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 117:
#line 318 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                     // Boolean constants
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 118:
#line 323 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Float & Double constants
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 119:
#line 330 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += " (" + *(yyvsp[-3]) + " " + *(yyvsp[-2]) + " " + *(yyvsp[-1]) + ")";
    delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 120:
#line 335 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 121:
#line 337 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 122:
#line 339 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 123:
#line 341 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 124:
#line 343 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 125:
#line 345 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 126:
#line 347 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 127:
#line 349 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 128:
#line 351 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 129:
#line 356 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 130:
#line 358 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 131:
#line 363 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { ;}
    break;

  case 132:
#line 363 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { ;}
    break;

  case 133:
#line 373 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
;}
    break;

  case 134:
#line 378 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = 0;
  ;}
    break;

  case 135:
#line 381 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0]) << "\n";
    delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 136:
#line 386 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "module asm " << " " << *(yyvsp[0]) << "\n";
  ;}
    break;

  case 137:
#line 389 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "implementation\n";
  ;}
    break;

  case 138:
#line 392 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 139:
#line 396 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 140:
#line 401 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {       // Function prototypes can be in const pool
    *O << *(yyvsp[0]) << "\n";
    delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 141:
#line 406 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  // Asm blocks can be in the const pool
    *O << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]); 
    (yyval) = 0;
  ;}
    break;

  case 142:
#line 411 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4]) << " " << *(yyvsp[-3]) << " " << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 143:
#line 416 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4]) << " " << *(yyvsp[-3]) << " " << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 144:
#line 421 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4]) << " " << *(yyvsp[-3]) << " " << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 145:
#line 426 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4]) << " " << *(yyvsp[-3]) << " " << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 146:
#line 431 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *O << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 147:
#line 436 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2]) << " = " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 148:
#line 441 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval) = 0;
  ;}
    break;

  case 152:
#line 451 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " = " + *(yyvsp[-1]);
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 153:
#line 456 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " = " + *(yyvsp[-1]);
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 154:
#line 461 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " = " + *(yyvsp[-1]);
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 155:
#line 466 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " = " + *(yyvsp[-1]);
    delete (yyvsp[-1]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 156:
#line 473 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1])->insert(0, "[ ");
    *(yyvsp[-1]) += " ]";
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 157:
#line 480 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += ", " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 159:
#line 486 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string();
  ;}
    break;

  case 163:
#line 495 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 164:
#line 497 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  (yyval) = (yyvsp[-1]);
  if (!(yyvsp[0])->empty())
    *(yyval) += " " + *(yyvsp[0]);
;}
    break;

  case 165:
#line 503 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += ", " + *(yyvsp[0]);
  ;}
    break;

  case 166:
#line 506 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = (yyvsp[0]);
  ;}
    break;

  case 167:
#line 510 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = (yyvsp[0]);
  ;}
    break;

  case 168:
#line 513 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += ", ...";
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 169:
#line 517 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = (yyvsp[0]);
  ;}
    break;

  case 170:
#line 520 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string();
  ;}
    break;

  case 171:
#line 525 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-7])->empty()) {
      (yyvsp[-6])->insert(0, *(yyvsp[-7]) + " ");
    }
    *(yyvsp[-6]) += " " + *(yyvsp[-5]) + "( " + *(yyvsp[-3]) + " )";
    if (!(yyvsp[-1])->empty()) {
      *(yyvsp[-6]) += " " + *(yyvsp[-1]);
    }
    if (!(yyvsp[0])->empty()) {
      *(yyvsp[-6]) += " " + *(yyvsp[0]);
    }
    (yyval) = (yyvsp[-6]);
  ;}
    break;

  case 172:
#line 539 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = new std::string("begin");
  ;}
    break;

  case 173:
#line 542 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval) = new std::string ("{");
  ;}
    break;

  case 174:
#line 546 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if (!(yyvsp[-2])->empty()) {
    *O << *(yyvsp[-2]) << " ";
  }
  *O << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
  delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
  (yyval) = 0;
;}
    break;

  case 175:
#line 555 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string("end"); ;}
    break;

  case 176:
#line 556 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string("}"); ;}
    break;

  case 177:
#line 558 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if ((yyvsp[-1]))
    *O << *(yyvsp[-1]);
  *O << '\n' << *(yyvsp[0]) << "\n";
;}
    break;

  case 181:
#line 570 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-2]) += " " + *(yyvsp[-1]) + " " + *(yyvsp[0]);
    delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 182:
#line 580 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 183:
#line 582 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 192:
#line 587 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1])->insert(0, "<");
    *(yyvsp[-1]) += ">";
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 194:
#line 593 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3])->empty()) {
      *(yyvsp[-4]) += " " + *(yyvsp[-3]);
    }
    *(yyvsp[-4]) += " " + *(yyvsp[-2]) + ", " + *(yyvsp[-1]);
    delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]);
    (yyval) = (yyvsp[-4]);
  ;}
    break;

  case 199:
#line 611 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 200:
#line 617 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 201:
#line 619 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Do not allow functions with 0 basic blocks   
  ;}
    break;

  case 202:
#line 626 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-1]) ;
  ;}
    break;

  case 203:
#line 630 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0]) << "\n";
    delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 204:
#line 635 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval) = 0;
  ;}
    break;

  case 205:
#line 638 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0]) << "\n";
    delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 206:
#line 644 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {              // Return with a result...
    *O << "    " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 207:
#line 649 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                       // Return with no result...
    *O << "    " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 208:
#line 654 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                         // Unconditional Branch...
    *O << "    " << *(yyvsp[-2]) << " " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 209:
#line 659 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  
    *O << "    " << *(yyvsp[-8]) << " " << *(yyvsp[-7]) << " " << *(yyvsp[-6]) << ", " << *(yyvsp[-4]) << " "
       << *(yyvsp[-3]) << ", " << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-8]); delete (yyvsp[-7]); delete (yyvsp[-6]); delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 210:
#line 665 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-8]) << " " << *(yyvsp[-7]) << " " << *(yyvsp[-6]) << ", " << *(yyvsp[-4]) << " " 
       << *(yyvsp[-3]) << " [" << *(yyvsp[-1]) << " ]\n";
    delete (yyvsp[-8]); delete (yyvsp[-7]); delete (yyvsp[-6]); delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-1]);
    (yyval) = 0;
  ;}
    break;

  case 211:
#line 671 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-7]) << " " << *(yyvsp[-6]) << " " << *(yyvsp[-5]) << ", " << *(yyvsp[-3]) << " " 
       << *(yyvsp[-2]) << "[]\n";
    delete (yyvsp[-7]); delete (yyvsp[-6]); delete (yyvsp[-5]); delete (yyvsp[-3]); delete (yyvsp[-2]);
    (yyval) = 0;
  ;}
    break;

  case 212:
#line 678 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-12]) << " " << *(yyvsp[-11]) << " " << *(yyvsp[-10]) << " " << *(yyvsp[-9]) << " ("
       << *(yyvsp[-7]) << ") " << *(yyvsp[-5]) << " " << *(yyvsp[-4]) << " " << *(yyvsp[-3]) << " " << *(yyvsp[-2]) << " "
       << *(yyvsp[-1]) << " " << *(yyvsp[0]) << "\n";
    delete (yyvsp[-12]); delete (yyvsp[-11]); delete (yyvsp[-10]); delete (yyvsp[-9]); delete (yyvsp[-7]); delete (yyvsp[-5]); delete (yyvsp[-4]);
    delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]); 
    (yyval) = 0;
  ;}
    break;

  case 213:
#line 686 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0]) << "\n";
    delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 214:
#line 691 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0]) << "\n";
    delete (yyvsp[0]);
    (yyval) = 0;
  ;}
    break;

  case 215:
#line 697 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += *(yyvsp[-4]) + " " + *(yyvsp[-3]) + ", " + *(yyvsp[-1]) + " " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 216:
#line 702 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4]) += *(yyvsp[-3]) + ", " + *(yyvsp[-1]) + " " + *(yyvsp[0]);
    delete (yyvsp[-3]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-4]);
  ;}
    break;

  case 217:
#line 709 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]); 
  ;}
    break;

  case 218:
#line 716 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {    // Used for PHI nodes
    *(yyvsp[-5]) += " [" + *(yyvsp[-3]) + "," + *(yyvsp[-1]) + "]";
    delete (yyvsp[-3]); delete (yyvsp[-1]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 219:
#line 721 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6]) += ", [" + *(yyvsp[-3]) + "," + *(yyvsp[-1]) + "]";
    delete (yyvsp[-3]); delete (yyvsp[-1]);
    (yyval) = (yyvsp[-6]);
  ;}
    break;

  case 221:
#line 730 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += ", " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 223:
#line 739 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 224:
#line 743 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 226:
#line 751 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4]) += " " + *(yyvsp[-3]) + " " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-4]);
  ;}
    break;

  case 227:
#line 756 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4]) += " " + *(yyvsp[-3]) + " " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-4]);
  ;}
    break;

  case 228:
#line 761 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4]) += " " + *(yyvsp[-3]) + " " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-3]); delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-4]);
  ;}
    break;

  case 229:
#line 766 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 230:
#line 771 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3]) += " " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 231:
#line 776 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3]) += " " + *(yyvsp[-2]) + " " + *(yyvsp[-1]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 232:
#line 781 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += " " + *(yyvsp[-4]) + ", " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 233:
#line 786 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3]) += " " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 234:
#line 791 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3]) += " " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 235:
#line 796 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += " " + *(yyvsp[-4]) + ", " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 236:
#line 801 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += " " + *(yyvsp[-4]) + ", " + *(yyvsp[-2]) + ", " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-2]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 237:
#line 806 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 238:
#line 811 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5])->empty())
      *(yyvsp[-6]) += " " + *(yyvsp[-5]);
    if (!(yyvsp[-6])->empty())
      *(yyvsp[-6]) += " ";
    *(yyvsp[-6]) += *(yyvsp[-4]) += " " + *(yyvsp[-3]) + "(" + *(yyvsp[-2]) + ")";
    delete (yyvsp[-5]); delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-1]);
    (yyval) = (yyvsp[-6]);
  ;}
    break;

  case 240:
#line 825 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[0])->insert(0, ", ");
    (yyval) = (yyvsp[0]);
  ;}
    break;

  case 241:
#line 829 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  (yyval) = new std::string(); ;}
    break;

  case 243:
#line 834 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval) = new std::string(); ;}
    break;

  case 244:
#line 837 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " " + *(yyvsp[-1]);
    if (!(yyvsp[0])->empty())
      *(yyvsp[-2]) += " " + *(yyvsp[0]);
    delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 245:
#line 844 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += " " + *(yyvsp[-4]) + ", " + *(yyvsp[-2]) + " " + *(yyvsp[-1]);
    if (!(yyvsp[0])->empty())
      *(yyvsp[-5]) += " " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 246:
#line 851 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2]) += " " + *(yyvsp[-1]);
    if (!(yyvsp[0])->empty())
      *(yyvsp[-2]) += " " + *(yyvsp[0]);
    delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-2]);
  ;}
    break;

  case 247:
#line 858 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5]) += " " + *(yyvsp[-4]) + ", " + *(yyvsp[-2]) + " " + *(yyvsp[-1]);
    if (!(yyvsp[0])->empty())
      *(yyvsp[-5]) += " " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 248:
#line 865 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1]) += " " + *(yyvsp[0]);
    delete (yyvsp[0]);
    (yyval) = (yyvsp[-1]);
  ;}
    break;

  case 249:
#line 870 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3])->empty())
      *(yyvsp[-3]) += " ";
    *(yyvsp[-3]) += *(yyvsp[-2]) + " " + *(yyvsp[-1]) + " " + *(yyvsp[0]);
    delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;

  case 250:
#line 877 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5])->empty())
      *(yyvsp[-5]) += " ";
    *(yyvsp[-5]) += *(yyvsp[-4]) + " " + *(yyvsp[-3]) + ", " + *(yyvsp[-1]) + " " + *(yyvsp[0]);
    delete (yyvsp[-4]); delete (yyvsp[-3]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-5]);
  ;}
    break;

  case 251:
#line 884 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3]) += *(yyvsp[-2]) + " " + *(yyvsp[-1]) + " " + *(yyvsp[0]);
    delete (yyvsp[-2]); delete (yyvsp[-1]); delete (yyvsp[0]);
    (yyval) = (yyvsp[-3]);
  ;}
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 3207 "UpgradeParser.tab.c"

  yyvsp -= yylen;
  yyssp -= yylen;


  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (YYPACT_NINF < yyn && yyn < YYLAST)
	{
	  int yytype = YYTRANSLATE (yychar);
	  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
	  YYSIZE_T yysize = yysize0;
	  YYSIZE_T yysize1;
	  int yysize_overflow = 0;
	  char *yymsg = 0;
#	  define YYERROR_VERBOSE_ARGS_MAXIMUM 5
	  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
	  int yyx;

#if 0
	  /* This is so xgettext sees the translatable formats that are
	     constructed on the fly.  */
	  YY_("syntax error, unexpected %s");
	  YY_("syntax error, unexpected %s, expecting %s");
	  YY_("syntax error, unexpected %s, expecting %s or %s");
	  YY_("syntax error, unexpected %s, expecting %s or %s or %s");
	  YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
#endif
	  char *yyfmt;
	  char const *yyf;
	  static char const yyunexpected[] = "syntax error, unexpected %s";
	  static char const yyexpecting[] = ", expecting %s";
	  static char const yyor[] = " or %s";
	  char yyformat[sizeof yyunexpected
			+ sizeof yyexpecting - 1
			+ ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
			   * (sizeof yyor - 1))];
	  char const *yyprefix = yyexpecting;

	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  int yyxbegin = yyn < 0 ? -yyn : 0;

	  /* Stay within bounds of both yycheck and yytname.  */
	  int yychecklim = YYLAST - yyn;
	  int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
	  int yycount = 1;

	  yyarg[0] = yytname[yytype];
	  yyfmt = yystpcpy (yyformat, yyunexpected);

	  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      {
		if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
		  {
		    yycount = 1;
		    yysize = yysize0;
		    yyformat[sizeof yyunexpected - 1] = '\0';
		    break;
		  }
		yyarg[yycount++] = yytname[yyx];
		yysize1 = yysize + yytnamerr (0, yytname[yyx]);
		yysize_overflow |= yysize1 < yysize;
		yysize = yysize1;
		yyfmt = yystpcpy (yyfmt, yyprefix);
		yyprefix = yyor;
	      }

	  yyf = YY_(yyformat);
	  yysize1 = yysize + yystrlen (yyf);
	  yysize_overflow |= yysize1 < yysize;
	  yysize = yysize1;

	  if (!yysize_overflow && yysize <= YYSTACK_ALLOC_MAXIMUM)
	    yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg)
	    {
	      /* Avoid sprintf, as that infringes on the user's name space.
		 Don't have undefined behavior even if the translation
		 produced a string with the wrong number of "%s"s.  */
	      char *yyp = yymsg;
	      int yyi = 0;
	      while ((*yyp = *yyf))
		{
		  if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		    {
		      yyp += yytnamerr (yyp, yyarg[yyi++]);
		      yyf += 2;
		    }
		  else
		    {
		      yyp++;
		      yyf++;
		    }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    {
	      yyerror (YY_("syntax error"));
	      goto yyexhaustedlab;
	    }
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror (YY_("syntax error"));
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
        {
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
        }
      else
	{
	  yydestruct ("Error: discarding", yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (0)
     goto yyerrorlab;

yyvsp -= yylen;
  yyssp -= yylen;
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping", yystos[yystate], yyvsp);
      YYPOPSTACK;
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token. */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK;
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 890 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"


int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << errMsg << '\n';
  exit(1);
}

