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
     TRUETOK = 263,
     FALSETOK = 264,
     NULL_TOK = 265,
     UNDEF = 266,
     ZEROINITIALIZER = 267,
     VOID = 268,
     BOOL = 269,
     SBYTE = 270,
     UBYTE = 271,
     SHORT = 272,
     USHORT = 273,
     INT = 274,
     UINT = 275,
     LONG = 276,
     ULONG = 277,
     FLOAT = 278,
     DOUBLE = 279,
     LABEL = 280,
     OPAQUE = 281,
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
     PHI_TOK = 357,
     SELECT = 358,
     SHL = 359,
     LSHR = 360,
     ASHR = 361,
     VAARG = 362,
     EXTRACTELEMENT = 363,
     INSERTELEMENT = 364,
     SHUFFLEVECTOR = 365,
     CAST = 366
   };
#endif
/* Tokens.  */
#define ESINT64VAL 258
#define EUINT64VAL 259
#define SINTVAL 260
#define UINTVAL 261
#define FPVAL 262
#define TRUETOK 263
#define FALSETOK 264
#define NULL_TOK 265
#define UNDEF 266
#define ZEROINITIALIZER 267
#define VOID 268
#define BOOL 269
#define SBYTE 270
#define UBYTE 271
#define SHORT 272
#define USHORT 273
#define INT 274
#define UINT 275
#define LONG 276
#define ULONG 277
#define FLOAT 278
#define DOUBLE 279
#define LABEL 280
#define OPAQUE 281
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
#define PHI_TOK 357
#define SELECT 358
#define SHL 359
#define LSHR 360
#define ASHR 361
#define VAARG 362
#define EXTRACTELEMENT 363
#define INSERTELEMENT 364
#define SHUFFLEVECTOR 365
#define CAST 366




/* Copy the first part of user declarations.  */
#line 14 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"

#include "ParserInternals.h"
#include <llvm/ADT/StringExtras.h>
#include <algorithm>
#include <list>
#include <utility>
#include <iostream>

#define YYERROR_VERBOSE 1
#define YYINCLUDED_STDLIB_H
#define YYDEBUG 1

int yylex();                       // declaration" of xxx warnings.
int yyparse();
extern int yydebug;

static std::string CurFilename;
static std::ostream *O = 0;
std::istream* LexInput = 0;
unsigned SizeOfPointer = 32;

void UpgradeAssembly(const std::string &infile, std::istream& in, 
                     std::ostream &out, bool debug)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  LexInput = &in;
  yydebug = debug;
  O = &out;

  if (yyparse()) {
    std::cerr << "Parse failed.\n";
    exit(1);
  }
}

const char* getCastOpcode(TypeInfo& SrcTy, TypeInfo&DstTy) {
  unsigned SrcBits = SrcTy.getBitWidth();
  unsigned DstBits = DstTy.getBitWidth();
  const char* opcode = "bitcast";
  // Run through the possibilities ...
  if (DstTy.isIntegral()) {                        // Casting to integral
    if (SrcTy.isIntegral()) {                      // Casting from integral
      if (DstBits < SrcBits)
        opcode = "trunc";
      else if (DstBits > SrcBits) {                // its an extension
        if (SrcTy.isSigned())
          opcode ="sext";                          // signed -> SEXT
        else
          opcode = "zext";                         // unsigned -> ZEXT
      } else {
        opcode = "bitcast";                        // Same size, No-op cast
      }
    } else if (SrcTy.isFloatingPoint()) {          // Casting from floating pt
      if (DstTy.isSigned()) 
        opcode = "fptosi";                         // FP -> sint
      else
        opcode = "fptoui";                         // FP -> uint 
    } else if (SrcTy.isPacked()) {
      assert(DstBits == SrcTy.getBitWidth() &&
               "Casting packed to integer of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(SrcTy.isPointer() &&
             "Casting from a value that is not first-class type");
      opcode = "ptrtoint";                         // ptr -> int
    }
  } else if (DstTy.isFloatingPoint()) {           // Casting to floating pt
    if (SrcTy.isIntegral()) {                     // Casting from integral
      if (SrcTy.isSigned())
        opcode = "sitofp";                         // sint -> FP
      else
        opcode = "uitofp";                         // uint -> FP
    } else if (SrcTy.isFloatingPoint()) {         // Casting from floating pt
      if (DstBits < SrcBits) {
        opcode = "fptrunc";                        // FP -> smaller FP
      } else if (DstBits > SrcBits) {
        opcode = "fpext";                          // FP -> larger FP
      } else  {
        opcode ="bitcast";                         // same size, no-op cast
      }
    } else if (SrcTy.isPacked()) {
      assert(DstBits == SrcTy.getBitWidth() &&
             "Casting packed to floating point of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(0 && "Casting pointer or non-first class to float");
    }
  } else if (DstTy.isPacked()) {
    if (SrcTy.isPacked()) {
      assert(DstTy.getBitWidth() == SrcTy.getBitWidth() &&
             "Casting packed to packed of different widths");
      opcode = "bitcast";                          // packed -> packed
    } else if (DstTy.getBitWidth() == SrcBits) {
      opcode = "bitcast";                          // float/int -> packed
    } else {
      assert(!"Illegal cast to packed (wrong type or size)");
    }
  } else if (DstTy.isPointer()) {
    if (SrcTy.isPointer()) {
      opcode = "bitcast";                          // ptr -> ptr
    } else if (SrcTy.isIntegral()) {
      opcode = "inttoptr";                         // int -> ptr
    } else {
      assert(!"Casting pointer to other than pointer or int");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }
  return opcode;
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
#line 130 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
typedef union YYSTYPE {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
} YYSTYPE;
/* Line 196 of yacc.c.  */
#line 436 "UpgradeParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 448 "UpgradeParser.tab.c"

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
#define YYLAST   1150

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  126
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  70
/* YYNRULES -- Number of rules. */
#define YYNRULES  249
/* YYNRULES -- Number of states. */
#define YYNSTATES  508

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   366

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     115,   116,   124,     2,   113,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     120,   112,   121,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   117,   114,   119,     2,     2,     2,     2,     2,   125,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     118,     2,     2,   122,     2,   123,     2,     2,     2,     2,
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
     105,   106,   107,   108,   109,   110,   111
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
      80,    81,    83,    85,    87,    89,    91,    93,    95,    96,
      98,   100,   102,   104,   106,   108,   111,   112,   113,   116,
     117,   121,   124,   125,   127,   128,   132,   134,   137,   139,
     141,   143,   145,   147,   149,   151,   153,   155,   157,   159,
     161,   163,   165,   167,   169,   171,   173,   175,   177,   180,
     185,   191,   197,   201,   204,   207,   209,   213,   215,   219,
     221,   222,   227,   231,   235,   240,   245,   249,   252,   255,
     258,   261,   264,   267,   270,   273,   276,   279,   286,   292,
     301,   308,   315,   322,   329,   336,   345,   354,   358,   360,
     362,   364,   366,   369,   372,   377,   380,   382,   387,   390,
     395,   402,   409,   416,   423,   427,   432,   433,   435,   437,
     439,   443,   447,   451,   455,   459,   463,   465,   466,   468,
     470,   472,   473,   476,   480,   482,   484,   488,   490,   491,
     500,   502,   504,   508,   510,   512,   516,   517,   519,   521,
     525,   526,   528,   530,   532,   534,   536,   538,   540,   542,
     544,   548,   550,   556,   558,   560,   562,   564,   567,   570,
     572,   575,   578,   579,   581,   584,   587,   591,   601,   611,
     620,   635,   637,   639,   646,   652,   655,   662,   670,   672,
     676,   678,   679,   682,   684,   690,   696,   702,   705,   710,
     715,   722,   727,   732,   739,   746,   749,   757,   759,   762,
     763,   765,   766,   770,   777,   781,   788,   791,   796,   803
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     157,     0,    -1,     5,    -1,     6,    -1,     3,    -1,     4,
      -1,    78,    -1,    79,    -1,    80,    -1,    81,    -1,    82,
      -1,    83,    -1,    84,    -1,    85,    -1,    86,    -1,    87,
      -1,    88,    -1,    89,    -1,    90,    -1,    91,    -1,    92,
      -1,    93,    -1,    94,    -1,    95,    -1,   104,    -1,   105,
      -1,   106,    -1,    21,    -1,    19,    -1,    17,    -1,    15,
      -1,    22,    -1,    20,    -1,    18,    -1,    16,    -1,   133,
      -1,   134,    -1,    23,    -1,    24,    -1,   165,   112,    -1,
      -1,    42,    -1,    43,    -1,    44,    -1,    48,    -1,    45,
      -1,    46,    -1,    47,    -1,    -1,    65,    -1,    66,    -1,
      67,    -1,    68,    -1,    69,    -1,    70,    -1,    64,     4,
      -1,    -1,    -1,    57,     4,    -1,    -1,   113,    57,     4,
      -1,    37,    30,    -1,    -1,   142,    -1,    -1,   113,   145,
     144,    -1,   142,    -1,    57,     4,    -1,   148,    -1,    13,
      -1,   150,    -1,    13,    -1,   150,    -1,    14,    -1,    15,
      -1,    16,    -1,    17,    -1,    18,    -1,    19,    -1,    20,
      -1,    21,    -1,    22,    -1,    23,    -1,    24,    -1,    25,
      -1,    26,    -1,   149,    -1,   179,    -1,   114,     4,    -1,
     147,   115,   152,   116,    -1,   117,     4,   118,   150,   119,
      -1,   120,     4,   118,   150,   121,    -1,   122,   151,   123,
      -1,   122,   123,    -1,   150,   124,    -1,   150,    -1,   151,
     113,   150,    -1,   151,    -1,   151,   113,    40,    -1,    40,
      -1,    -1,   148,   117,   155,   119,    -1,   148,   117,   119,
      -1,   148,   125,    30,    -1,   148,   120,   155,   121,    -1,
     148,   122,   155,   123,    -1,   148,   122,   123,    -1,   148,
      10,    -1,   148,    11,    -1,   148,   179,    -1,   148,   154,
      -1,   148,    12,    -1,   133,   128,    -1,   134,     4,    -1,
      14,     8,    -1,    14,     9,    -1,   136,     7,    -1,   111,
     115,   153,    39,   148,   116,    -1,   101,   115,   153,   193,
     116,    -1,   103,   115,   153,   113,   153,   113,   153,   116,
      -1,   129,   115,   153,   113,   153,   116,    -1,   130,   115,
     153,   113,   153,   116,    -1,   131,   115,   153,   113,   153,
     116,    -1,   132,   115,   153,   113,   153,   116,    -1,   108,
     115,   153,   113,   153,   116,    -1,   109,   115,   153,   113,
     153,   113,   153,   116,    -1,   110,   115,   153,   113,   153,
     113,   153,   116,    -1,   155,   113,   153,    -1,   153,    -1,
      35,    -1,    36,    -1,   158,    -1,   158,   174,    -1,   158,
     176,    -1,   158,    62,    61,   160,    -1,   158,    31,    -1,
     159,    -1,   159,   137,    27,   146,    -1,   159,   176,    -1,
     159,    62,    61,   160,    -1,   159,   137,   138,   156,   153,
     144,    -1,   159,   137,    50,   156,   148,   144,    -1,   159,
     137,    45,   156,   148,   144,    -1,   159,   137,    47,   156,
     148,   144,    -1,   159,    51,   162,    -1,   159,    58,   112,
     163,    -1,    -1,    30,    -1,    56,    -1,    55,    -1,    53,
     112,   161,    -1,    54,   112,     4,    -1,    52,   112,    30,
      -1,    71,   112,    30,    -1,   117,   164,   119,    -1,   164,
     113,    30,    -1,    30,    -1,    -1,    28,    -1,    30,    -1,
     165,    -1,    -1,   148,   166,    -1,   168,   113,   167,    -1,
     167,    -1,   168,    -1,   168,   113,    40,    -1,    40,    -1,
      -1,   139,   146,   165,   115,   169,   116,   143,   140,    -1,
      32,    -1,   122,    -1,   138,   170,   171,    -1,    33,    -1,
     123,    -1,   172,   182,   173,    -1,    -1,    45,    -1,    47,
      -1,    34,   175,   170,    -1,    -1,    63,    -1,     3,    -1,
       4,    -1,     7,    -1,     8,    -1,     9,    -1,    10,    -1,
      11,    -1,    12,    -1,   120,   155,   121,    -1,   154,    -1,
      61,   177,    30,   113,    30,    -1,   127,    -1,   165,    -1,
     179,    -1,   178,    -1,   148,   180,    -1,   182,   183,    -1,
     183,    -1,   184,   185,    -1,   184,   187,    -1,    -1,    29,
      -1,    72,   181,    -1,    72,    13,    -1,    73,    25,   180,
      -1,    73,    14,   180,   113,    25,   180,   113,    25,   180,
      -1,    74,   135,   180,   113,    25,   180,   117,   186,   119,
      -1,    74,   135,   180,   113,    25,   180,   117,   119,    -1,
     137,    75,   139,   146,   180,   115,   190,   116,    39,    25,
     180,    76,    25,   180,    -1,    76,    -1,    77,    -1,   186,
     135,   178,   113,    25,   180,    -1,   135,   178,   113,    25,
     180,    -1,   137,   192,    -1,   148,   117,   180,   113,   180,
     119,    -1,   188,   113,   117,   180,   113,   180,   119,    -1,
     181,    -1,   189,   113,   181,    -1,   189,    -1,    -1,    60,
      59,    -1,    59,    -1,   129,   148,   180,   113,   180,    -1,
     130,   148,   180,   113,   180,    -1,   131,   148,   180,   113,
     180,    -1,    49,   181,    -1,   132,   181,   113,   181,    -1,
     111,   181,    39,   148,    -1,   103,   181,   113,   181,   113,
     181,    -1,   107,   181,   113,   148,    -1,   108,   181,   113,
     181,    -1,   109,   181,   113,   181,   113,   181,    -1,   110,
     181,   113,   181,   113,   181,    -1,   102,   188,    -1,   191,
     139,   146,   180,   115,   190,   116,    -1,   195,    -1,   113,
     189,    -1,    -1,    38,    -1,    -1,    96,   148,   141,    -1,
      96,   148,   113,    20,   180,   141,    -1,    97,   148,   141,
      -1,    97,   148,   113,    20,   180,   141,    -1,    98,   181,
      -1,   194,    99,   148,   180,    -1,   194,   100,   181,   113,
     148,   180,    -1,   101,   148,   180,   193,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   189,   189,   189,   190,   190,   194,   194,   194,   194,
     194,   194,   194,   194,   194,   195,   195,   195,   196,   196,
     196,   196,   196,   196,   197,   197,   197,   201,   201,   201,
     201,   202,   202,   202,   202,   203,   203,   204,   204,   207,
     211,   216,   216,   216,   216,   216,   216,   217,   218,   221,
     221,   221,   221,   221,   222,   223,   228,   233,   234,   237,
     238,   246,   252,   253,   256,   257,   266,   267,   280,   280,
     281,   281,   282,   286,   286,   286,   286,   286,   286,   286,
     287,   287,   287,   287,   287,   288,   288,   289,   295,   300,
     306,   313,   320,   326,   330,   340,   343,   351,   352,   357,
     360,   370,   376,   381,   387,   393,   399,   404,   410,   416,
     422,   428,   434,   440,   446,   452,   458,   466,   473,   479,
     484,   489,   494,   499,   504,   509,   514,   524,   529,   534,
     534,   544,   549,   552,   557,   561,   565,   568,   573,   578,
     583,   589,   595,   601,   607,   612,   617,   622,   624,   624,
     627,   632,   639,   644,   651,   658,   663,   664,   672,   672,
     673,   673,   675,   682,   686,   690,   693,   698,   701,   703,
     723,   726,   730,   739,   740,   742,   750,   751,   752,   756,
     769,   770,   773,   774,   775,   776,   777,   778,   779,   780,
     781,   786,   787,   796,   796,   799,   799,   805,   812,   814,
     821,   825,   830,   833,   839,   844,   849,   854,   861,   867,
     873,   886,   891,   897,   902,   910,   917,   923,   931,   932,
     940,   941,   945,   950,   953,   958,   963,   968,   973,   978,
     985,   990,   995,  1000,  1005,  1010,  1015,  1024,  1029,  1033,
    1037,  1038,  1041,  1048,  1055,  1062,  1069,  1074,  1081,  1088
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "ESINT64VAL", "EUINT64VAL", "SINTVAL",
  "UINTVAL", "FPVAL", "TRUETOK", "FALSETOK", "NULL_TOK", "UNDEF",
  "ZEROINITIALIZER", "VOID", "BOOL", "SBYTE", "UBYTE", "SHORT", "USHORT",
  "INT", "UINT", "LONG", "ULONG", "FLOAT", "DOUBLE", "LABEL", "OPAQUE",
  "TYPE", "VAR_ID", "LABELSTR", "STRINGCONSTANT", "IMPLEMENTATION",
  "BEGINTOK", "ENDTOK", "DECLARE", "GLOBAL", "CONSTANT", "SECTION",
  "VOLATILE", "TO", "DOTDOTDOT", "CONST", "INTERNAL", "LINKONCE", "WEAK",
  "DLLIMPORT", "DLLEXPORT", "EXTERN_WEAK", "APPENDING", "NOT", "EXTERNAL",
  "TARGET", "TRIPLE", "ENDIAN", "POINTERSIZE", "LITTLE", "BIG", "ALIGN",
  "DEPLIBS", "CALL", "TAIL", "ASM_TOK", "MODULE", "SIDEEFFECT", "CC_TOK",
  "CCC_TOK", "CSRETCC_TOK", "FASTCC_TOK", "COLDCC_TOK",
  "X86_STDCALLCC_TOK", "X86_FASTCALLCC_TOK", "DATALAYOUT", "RET", "BR",
  "SWITCH", "INVOKE", "UNWIND", "UNREACHABLE", "ADD", "SUB", "MUL", "UDIV",
  "SDIV", "FDIV", "UREM", "SREM", "FREM", "AND", "OR", "XOR", "SETLE",
  "SETGE", "SETLT", "SETGT", "SETEQ", "SETNE", "MALLOC", "ALLOCA", "FREE",
  "LOAD", "STORE", "GETELEMENTPTR", "PHI_TOK", "SELECT", "SHL", "LSHR",
  "ASHR", "VAARG", "EXTRACTELEMENT", "INSERTELEMENT", "SHUFFLEVECTOR",
  "CAST", "'='", "','", "'\\\\'", "'('", "')'", "'['", "'x'", "']'", "'<'",
  "'>'", "'{'", "'}'", "'*'", "'c'", "$accept", "IntVal", "EInt64Val",
  "ArithmeticOps", "LogicalOps", "SetCondOps", "ShiftOps", "SIntType",
  "UIntType", "IntType", "FPType", "OptAssign", "OptLinkage",
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
     365,   366,    61,    44,    92,    40,    41,    91,   120,    93,
      60,    62,   123,   125,    42,    99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   126,   127,   127,   128,   128,   129,   129,   129,   129,
     129,   129,   129,   129,   129,   130,   130,   130,   131,   131,
     131,   131,   131,   131,   132,   132,   132,   133,   133,   133,
     133,   134,   134,   134,   134,   135,   135,   136,   136,   137,
     137,   138,   138,   138,   138,   138,   138,   138,   138,   139,
     139,   139,   139,   139,   139,   139,   139,   140,   140,   141,
     141,   142,   143,   143,   144,   144,   145,   145,   146,   146,
     147,   147,   148,   149,   149,   149,   149,   149,   149,   149,
     149,   149,   149,   149,   149,   150,   150,   150,   150,   150,
     150,   150,   150,   150,   150,   151,   151,   152,   152,   152,
     152,   153,   153,   153,   153,   153,   153,   153,   153,   153,
     153,   153,   153,   153,   153,   153,   153,   154,   154,   154,
     154,   154,   154,   154,   154,   154,   154,   155,   155,   156,
     156,   157,   158,   158,   158,   158,   158,   159,   159,   159,
     159,   159,   159,   159,   159,   159,   159,   160,   161,   161,
     162,   162,   162,   162,   163,   164,   164,   164,   165,   165,
     166,   166,   167,   168,   168,   169,   169,   169,   169,   170,
     171,   171,   172,   173,   173,   174,   175,   175,   175,   176,
     177,   177,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   179,   179,   180,   180,   181,   182,   182,
     183,   184,   184,   184,   185,   185,   185,   185,   185,   185,
     185,   185,   185,   186,   186,   187,   188,   188,   189,   189,
     190,   190,   191,   191,   192,   192,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   193,   193,
     194,   194,   195,   195,   195,   195,   195,   195,   195,   195
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       0,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       1,     1,     1,     1,     1,     2,     0,     0,     2,     0,
       3,     2,     0,     1,     0,     3,     1,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     4,
       5,     5,     3,     2,     2,     1,     3,     1,     3,     1,
       0,     4,     3,     3,     4,     4,     3,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     6,     5,     8,
       6,     6,     6,     6,     6,     8,     8,     3,     1,     1,
       1,     1,     2,     2,     4,     2,     1,     4,     2,     4,
       6,     6,     6,     6,     3,     4,     0,     1,     1,     1,
       3,     3,     3,     3,     3,     3,     1,     0,     1,     1,
       1,     0,     2,     3,     1,     1,     3,     1,     0,     8,
       1,     1,     3,     1,     1,     3,     0,     1,     1,     3,
       0,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     1,     5,     1,     1,     1,     1,     2,     2,     1,
       2,     2,     0,     1,     2,     2,     3,     9,     9,     8,
      14,     1,     1,     6,     5,     2,     6,     7,     1,     3,
       1,     0,     2,     1,     5,     5,     5,     2,     4,     4,
       6,     4,     4,     6,     6,     2,     7,     1,     2,     0,
       1,     0,     3,     6,     3,     6,     2,     4,     6,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
     146,     0,    48,   136,     1,   135,   176,    41,    42,    43,
      45,    46,    47,    44,     0,    56,   202,   132,   133,   158,
     159,     0,     0,     0,    48,     0,   138,   177,   178,    56,
       0,     0,    49,    50,    51,    52,    53,    54,     0,     0,
     203,   202,   199,    40,     0,     0,     0,     0,   144,     0,
       0,     0,     0,     0,     0,     0,    39,   179,   147,   134,
      55,     2,     3,    69,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,     0,     0,     0,
       0,   193,     0,     0,    68,    86,    72,   194,    87,   170,
     171,   172,   173,   174,   175,   198,     0,     0,     0,   211,
     212,   241,   200,   201,     0,     0,     0,     0,   157,   145,
     139,   137,   129,   130,     0,     0,     0,     0,    88,     0,
       0,    71,    93,    95,     0,     0,   100,    94,   205,     0,
     204,     0,     0,    30,    34,    29,    33,    28,    32,    27,
      31,    35,    36,     0,   240,     0,   223,     0,    56,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,     0,     0,     0,
       0,     0,     0,    24,    25,    26,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    56,   215,     0,   237,   152,
     149,   148,   150,   151,   153,   156,     0,    64,    64,    64,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,     0,     0,     0,     0,    64,     0,     0,     0,    92,
     168,    99,    97,     0,   182,   183,   184,   185,   186,   187,
     188,   189,   180,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   191,   196,   195,   197,     0,   206,
       0,   227,   222,     0,    59,    59,   246,     0,     0,   235,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   154,     0,   142,   143,   141,   114,
     115,     4,     5,   112,   113,   116,   107,   108,   111,     0,
       0,     0,     0,   110,   109,   140,    70,    70,    96,   167,
     161,   164,   165,     0,     0,    89,   181,     0,     0,     0,
       0,     0,     0,     0,   128,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   242,     0,   244,   239,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   155,     0,     0,    66,    64,   102,     0,
       0,   106,     0,   103,    90,    91,   160,   162,     0,    62,
      98,     0,   239,     0,     0,     0,     0,     0,     0,   190,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   249,     0,     0,     0,   231,   232,     0,     0,   229,
       0,     0,     0,   228,     0,   247,     0,    61,    67,    65,
     101,   104,   105,   166,   163,    63,    57,     0,     0,     0,
       0,     0,     0,     0,   127,     0,     0,     0,     0,     0,
       0,   221,    59,    60,    59,   218,   238,     0,     0,     0,
       0,     0,   224,   225,   226,   221,     0,     0,   169,   192,
     118,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   220,     0,     0,   243,   245,     0,     0,     0,
     230,   233,   234,     0,   248,    58,     0,   124,     0,     0,
     117,   120,   121,   122,   123,     0,   209,     0,     0,     0,
     219,   216,     0,   236,     0,     0,     0,   207,     0,   208,
       0,     0,   217,   119,   125,   126,     0,     0,     0,     0,
       0,     0,   214,     0,     0,   213,     0,   210
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,    81,   283,   240,   241,   242,   243,   211,   212,   143,
     213,    24,    15,    38,   438,   324,   346,   406,   276,   347,
      82,    83,   214,    85,    86,   124,   223,   314,   244,   315,
     114,     1,     2,     3,    59,   192,    48,   109,   196,    87,
     357,   301,   302,   303,    39,    91,    16,    94,    17,    29,
      18,   307,   245,    88,   247,   425,    41,    42,    43,   102,
     478,   103,   259,   452,   453,   185,   186,   381,   187,   188
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -439
static const short int yypact[] =
{
    -439,    63,    22,   577,  -439,  -439,     2,  -439,  -439,  -439,
    -439,  -439,  -439,  -439,    37,   188,    72,  -439,  -439,  -439,
    -439,    21,    -8,    45,   150,     9,  -439,  -439,  -439,   188,
      94,   132,  -439,  -439,  -439,  -439,  -439,  -439,   883,   -27,
    -439,   -23,  -439,    14,    41,    50,    56,    57,  -439,    53,
      94,   883,    74,    74,    74,    74,  -439,  -439,  -439,  -439,
    -439,  -439,  -439,    59,  -439,  -439,  -439,  -439,  -439,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,  -439,   168,   171,   174,
     457,  -439,    66,    65,  -439,  -439,   -53,  -439,  -439,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,   935,    34,   224,  -439,
    -439,  1039,  -439,  -439,   151,    64,   184,   159,   160,  -439,
    -439,  -439,  -439,  -439,   965,   965,   965,   993,  -439,    81,
      83,  -439,  -439,   -53,   -72,    87,   742,  -439,    59,   588,
    -439,   588,   588,  -439,  -439,  -439,  -439,  -439,  -439,  -439,
    -439,  -439,  -439,   588,  -439,   965,  -439,   158,   188,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,  -439,  -439,  -439,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,  -439,   965,   965,   965,
     965,   965,   965,  -439,  -439,  -439,   965,   965,   965,   965,
     965,   965,   965,   965,   965,   188,  -439,    23,  -439,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,   -34,    97,    97,    97,
     119,   147,   212,   152,   214,   155,   216,   157,   217,   215,
     220,   163,   219,   221,   429,    97,   965,   965,   965,  -439,
     770,  -439,   111,   113,  -439,  -439,  -439,  -439,  -439,  -439,
    -439,  -439,   167,   116,   121,   133,   134,   135,   144,   993,
     149,   153,   154,   162,  -439,  -439,  -439,  -439,   165,  -439,
     169,  -439,  -439,   883,   170,   172,  -439,   588,   130,   173,
     177,   178,   185,   186,   189,   226,   588,   588,   588,   190,
     883,   965,   965,   236,  -439,    -7,  -439,  -439,  -439,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,  -439,  -439,  -439,   798,
     993,   542,   237,  -439,  -439,  -439,   -22,   -16,   -53,  -439,
      66,  -439,   195,   193,   825,  -439,  -439,   240,   993,   993,
     993,   993,   993,   993,  -439,   -41,   993,   993,   993,   993,
     248,   250,   588,     3,  -439,     4,  -439,   201,   588,   164,
     965,   965,   965,   965,   965,   965,   202,   204,   205,   965,
     588,   588,   206,  -439,   254,   317,  -439,    97,  -439,   -20,
     -32,  -439,   -68,  -439,  -439,  -439,  -439,  -439,   855,   286,
    -439,   213,   201,   218,   223,   225,   230,   288,   993,  -439,
     231,   232,   234,   235,   588,   588,   210,   588,   324,   588,
     965,  -439,   238,   588,   239,  -439,  -439,   244,   246,  -439,
     588,   588,   588,  -439,   245,  -439,   965,  -439,  -439,  -439,
    -439,  -439,  -439,  -439,  -439,  -439,   272,   300,   233,   993,
     993,   993,   993,   965,  -439,   993,   993,   993,   993,   249,
     253,   965,   252,  -439,   252,  -439,   261,   588,   263,   965,
     965,   965,  -439,  -439,  -439,   965,   588,   328,  -439,  -439,
    -439,   265,   251,   266,   267,   271,   273,   274,   278,   282,
     325,    -1,   261,   283,   296,  -439,  -439,   965,   247,   588,
    -439,  -439,  -439,   285,  -439,  -439,   993,  -439,   993,   993,
    -439,  -439,  -439,  -439,  -439,   588,  -439,   634,    16,   349,
    -439,  -439,   287,  -439,   289,   291,   292,  -439,   297,  -439,
     634,   377,  -439,  -439,  -439,  -439,   379,   298,   588,   588,
     384,   336,  -439,   588,   388,  -439,   588,  -439
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -439,  -439,  -439,   313,   314,   315,   316,   -97,   -96,  -424,
    -439,   375,   396,  -102,  -439,  -251,    67,  -439,  -190,  -439,
     -44,  -439,   -38,  -439,   -69,   299,  -439,  -105,   208,  -177,
      62,  -439,  -439,  -439,   373,  -439,  -439,  -439,  -439,     0,
    -439,    69,  -439,  -439,   395,  -439,  -439,  -439,  -439,  -439,
     425,  -439,  -438,  -103,    -6,     7,  -439,   390,  -439,  -439,
    -439,  -439,  -439,    49,    -3,  -439,  -439,    71,  -439,  -439
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -132
static const short int yytable[] =
{
      84,   141,   142,    25,   326,    89,    40,   111,   277,   278,
      92,   123,   215,    84,   133,   134,   135,   136,   137,   138,
     139,   140,  -131,   377,   379,   295,   246,   477,   246,   246,
     344,   133,   134,   135,   136,   137,   138,   139,   140,   488,
     246,   218,    19,    25,    20,   368,   253,    27,   131,    28,
     345,   219,   497,     5,   490,   402,     6,   123,   129,   132,
     378,   378,   -70,     4,     7,     8,     9,    10,    11,    12,
      13,   127,   368,    44,    45,    46,   197,   198,   199,   273,
     369,   368,   125,   270,    14,   274,    96,    97,    98,   401,
      99,   100,    47,   368,    19,    90,    20,   354,    30,   400,
      93,    40,   127,   130,    49,   355,    50,   129,   127,   112,
     113,   294,   349,   350,   352,   115,   116,   117,   476,   190,
     191,    56,   271,   272,    58,   248,   249,   279,   280,   254,
     255,   129,   257,   258,   129,   489,    60,   250,   129,   129,
     129,   129,   129,   266,   267,   268,   129,   296,   297,   298,
     -30,   -30,   251,   104,   246,   -29,   -29,   399,   -28,   -28,
     -27,   -27,   105,   246,   246,   246,   281,   282,   106,   107,
     108,   455,   118,   456,   -71,   119,   256,    51,   120,   260,
     126,   189,   300,   261,   262,   263,   264,   265,   193,   194,
     195,   269,     7,     8,     9,    52,    11,    53,    13,   216,
      54,   217,   220,   362,   363,   364,   365,   366,   367,   322,
     275,   370,   371,   372,   373,    84,   -34,   252,   -33,   246,
     -32,   -31,   -37,   284,   304,   246,   340,   -38,   285,   305,
     306,   308,    84,   341,   129,   298,   309,   246,   246,   133,
     134,   135,   136,   137,   138,   139,   140,   328,   310,   311,
     312,   327,    31,    32,    33,    34,    35,    36,    37,   313,
     336,   337,   338,   414,   316,   335,   343,   353,   317,   318,
     361,   246,   246,   374,   246,   375,   246,   319,   320,   342,
     246,   383,   321,   323,   397,   325,   329,   246,   246,   246,
     330,   331,   129,   385,   129,   129,   129,   389,   332,   333,
     356,   129,   334,   339,   441,   442,   443,   444,   358,   359,
     446,   447,   448,   449,   380,   390,   376,   391,   392,   396,
     300,   398,   382,   344,   246,   421,   407,   413,   423,   437,
     439,   409,   465,   246,   394,   395,   410,   384,   411,   386,
     387,   388,   129,   412,   415,   416,   393,   417,   418,   440,
     475,   427,   429,   378,   141,   142,   246,   430,   436,   431,
     435,   484,   450,   485,   486,   454,   481,   467,   419,   420,
     451,   422,   246,   424,   457,   445,   459,   428,   466,   468,
     469,   141,   142,   129,   432,   433,   434,   470,   491,   471,
     472,   129,   129,   129,   473,   246,   246,   129,   474,   479,
     246,   483,   498,   246,   499,   493,   492,   494,   495,   503,
     496,   500,   504,   506,   181,   182,   183,   184,   101,   129,
      55,   458,   293,   110,    57,   222,   405,   404,    26,   426,
     464,    95,   463,   408,    61,    62,   460,   461,   462,   286,
     287,   288,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   482,     0,     0,     0,    19,     0,    20,
       0,     0,    61,    62,   480,     0,     0,     0,     0,   487,
     121,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,     0,    19,     0,    20,     0,     0,
       0,     0,   501,   502,     0,     0,     0,   505,     0,     0,
     507,     0,     0,     0,     0,     0,     0,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,     0,     0,     0,     0,     0,
     233,     0,   234,   173,   174,   175,     0,   235,   236,   237,
     238,     0,     0,     0,     0,     0,   289,    61,    62,   290,
       0,   291,     0,     0,   292,   121,   200,   201,   202,   203,
     204,   205,   206,   207,   208,   209,   210,    75,    76,     0,
      19,    77,    20,     0,    78,     0,     0,    79,     0,    80,
     122,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   224,   225,    61,    62,   226,   227,   228,   229,   230,
     231,     0,     0,     0,   -40,    19,     0,    20,     0,     0,
       0,     6,   -40,   -40,     0,     0,    19,     0,    20,   -40,
     -40,   -40,   -40,   -40,   -40,   -40,     0,   -40,    21,     0,
       0,     0,     0,     0,     0,    22,     0,   224,   225,    23,
       0,   226,   227,   228,   229,   230,   231,     0,     0,   232,
       0,     0,     0,     0,     0,     0,    77,     0,     0,    78,
       0,     0,    79,     0,    80,   351,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,     0,     0,     0,     0,     0,   233,
       0,   234,   173,   174,   175,   232,   235,   236,   237,   238,
       0,     0,     0,     0,     0,     0,     0,     0,   239,     0,
       0,     0,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
       0,     0,     0,     0,     0,   233,     0,   234,   173,   174,
     175,     0,   235,   236,   237,   238,     0,    61,    62,     0,
       0,     0,     0,     0,   239,   121,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,     0,
      19,     0,    20,     0,     0,    61,    62,     0,     0,     0,
       0,     0,   221,   121,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,     0,    19,     0,
      20,     0,     0,    61,    62,     0,     0,     0,     0,     0,
     299,   121,   200,   201,   202,   203,   204,   205,   206,   207,
     208,   209,   210,    75,    76,     0,    19,     0,    20,     0,
      61,    62,     0,     0,     0,     0,     0,     0,   121,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,     0,    19,     0,    20,    77,     0,     0,    78,
      61,    62,    79,     0,    80,   360,     0,     0,   121,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,     0,    19,    77,    20,     0,    78,    61,    62,
      79,     0,    80,     0,     0,   403,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
       0,    19,    77,    20,     0,    78,     0,   348,    79,     0,
      80,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    77,
      61,    62,    78,     0,     0,    79,     0,    80,   128,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,     0,    19,     0,    20,     0,     0,     0,    77,
      61,    62,    78,     0,     0,    79,     0,    80,   121,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,     0,    19,     0,    20,     0,    77,    61,    62,
      78,     0,     0,    79,     0,    80,   121,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,    75,    76,
       0,    19,     0,    20,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    77,
       0,     0,    78,     0,     0,    79,     0,    80,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   144,     0,    77,
       0,     0,    78,     0,     0,    79,     0,    80,   145,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   146,   147,
       0,     0,     0,     0,     0,     0,     0,    77,     0,     0,
      78,     0,     0,    79,   148,    80,     0,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,   167,   168,   169,     0,     0,
     170,   171,   172,   173,   174,   175,   176,   177,   178,   179,
     180
};

static const short int yycheck[] =
{
      38,    98,    98,     3,   255,    32,    29,    51,   198,   199,
      33,    80,   117,    51,    15,    16,    17,    18,    19,    20,
      21,    22,     0,    20,    20,   215,   129,   451,   131,   132,
      37,    15,    16,    17,    18,    19,    20,    21,    22,   477,
     143,   113,    28,    43,    30,   113,   148,    45,    14,    47,
      57,   123,   490,    31,   478,   123,    34,   126,    96,    25,
      57,    57,   115,     0,    42,    43,    44,    45,    46,    47,
      48,   124,   113,    52,    53,    54,   114,   115,   116,   113,
     121,   113,    82,   185,    62,   119,    72,    73,    74,   121,
      76,    77,    71,   113,    28,   122,    30,   119,    61,   119,
     123,    29,   124,    96,   112,   121,    61,   145,   124,    35,
      36,   214,   289,   290,   291,    53,    54,    55,   119,    55,
      56,   112,    99,   100,    30,   131,   132,     8,     9,   167,
     168,   169,   170,   171,   172,   119,     4,   143,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   216,   217,   218,
       3,     4,   145,   112,   257,     3,     4,   347,     3,     4,
       3,     4,   112,   266,   267,   268,     3,     4,   112,   112,
     117,   422,     4,   424,   115,     4,   169,    27,     4,   172,
     115,    30,   220,   176,   177,   178,   179,   180,     4,    30,
      30,   184,    42,    43,    44,    45,    46,    47,    48,   118,
      50,   118,   115,   308,   309,   310,   311,   312,   313,   253,
     113,   316,   317,   318,   319,   253,     4,    59,     4,   322,
       4,     4,     7,     4,   113,   328,   270,     7,     7,   116,
      63,   115,   270,   271,   272,   304,   115,   340,   341,    15,
      16,    17,    18,    19,    20,    21,    22,   117,   115,   115,
     115,   257,    64,    65,    66,    67,    68,    69,    70,   115,
     266,   267,   268,   368,   115,    39,    30,    30,   115,   115,
      30,   374,   375,    25,   377,    25,   379,   115,   113,   272,
     383,   117,   113,   113,    30,   113,   113,   390,   391,   392,
     113,   113,   330,   331,   332,   333,   334,   335,   113,   113,
     300,   339,   113,   113,   409,   410,   411,   412,   113,   116,
     415,   416,   417,   418,   113,   113,   322,   113,   113,   113,
     358,     4,   328,    37,   427,   115,   113,    39,     4,    57,
      30,   113,     4,   436,   340,   341,   113,   330,   113,   332,
     333,   334,   380,   113,   113,   113,   339,   113,   113,   116,
      25,   113,   113,    57,   451,   451,   459,   113,   396,   113,
     115,   466,   113,   468,   469,   113,   119,   116,   374,   375,
     117,   377,   475,   379,   113,   413,   113,   383,   113,   113,
     113,   478,   478,   421,   390,   391,   392,   116,    39,   116,
     116,   429,   430,   431,   116,   498,   499,   435,   116,   116,
     503,   116,    25,   506,    25,   116,   119,   116,   116,    25,
     113,   113,    76,    25,   101,   101,   101,   101,    43,   457,
      24,   427,   214,    50,    29,   126,   359,   358,     3,   380,
     436,    41,   435,   362,     5,     6,   429,   430,   431,    10,
      11,    12,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   459,    -1,    -1,    -1,    28,    -1,    30,
      -1,    -1,     5,     6,   457,    -1,    -1,    -1,    -1,   475,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    -1,    28,    -1,    30,    -1,    -1,
      -1,    -1,   498,   499,    -1,    -1,    -1,   503,    -1,    -1,
     506,    -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    -1,    -1,    -1,    -1,    -1,
     101,    -1,   103,   104,   105,   106,    -1,   108,   109,   110,
     111,    -1,    -1,    -1,    -1,    -1,   117,     5,     6,   120,
      -1,   122,    -1,    -1,   125,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    -1,
      28,   114,    30,    -1,   117,    -1,    -1,   120,    -1,   122,
     123,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    -1,    -1,    -1,    27,    28,    -1,    30,    -1,    -1,
      -1,    34,    35,    36,    -1,    -1,    28,    -1,    30,    42,
      43,    44,    45,    46,    47,    48,    -1,    50,    51,    -1,
      -1,    -1,    -1,    -1,    -1,    58,    -1,     3,     4,    62,
      -1,     7,     8,     9,    10,    11,    12,    -1,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,   114,    -1,    -1,   117,
      -1,    -1,   120,    -1,   122,   123,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    -1,    -1,    -1,    -1,    -1,   101,
      -1,   103,   104,   105,   106,    61,   108,   109,   110,   111,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   120,    -1,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      -1,    -1,    -1,    -1,    -1,   101,    -1,   103,   104,   105,
     106,    -1,   108,   109,   110,   111,    -1,     5,     6,    -1,
      -1,    -1,    -1,    -1,   120,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    -1,
      28,    -1,    30,    -1,    -1,     5,     6,    -1,    -1,    -1,
      -1,    -1,    40,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    -1,    28,    -1,
      30,    -1,    -1,     5,     6,    -1,    -1,    -1,    -1,    -1,
      40,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    -1,    28,    -1,    30,    -1,
       5,     6,    -1,    -1,    -1,    -1,    -1,    -1,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    -1,    28,    -1,    30,   114,    -1,    -1,   117,
       5,     6,   120,    -1,   122,    40,    -1,    -1,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    -1,    28,   114,    30,    -1,   117,     5,     6,
     120,    -1,   122,    -1,    -1,    40,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      -1,    28,   114,    30,    -1,   117,    -1,   119,   120,    -1,
     122,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,
       5,     6,   117,    -1,    -1,   120,    -1,   122,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    -1,    28,    -1,    30,    -1,    -1,    -1,   114,
       5,     6,   117,    -1,    -1,   120,    -1,   122,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    -1,    28,    -1,    30,    -1,   114,     5,     6,
     117,    -1,    -1,   120,    -1,   122,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      -1,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,
      -1,    -1,   117,    -1,    -1,   120,    -1,   122,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,    -1,   114,
      -1,    -1,   117,    -1,    -1,   120,    -1,   122,    49,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    59,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   114,    -1,    -1,
     117,    -1,    -1,   120,    75,   122,    -1,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    -1,    -1,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   157,   158,   159,     0,    31,    34,    42,    43,    44,
      45,    46,    47,    48,    62,   138,   172,   174,   176,    28,
      30,    51,    58,    62,   137,   165,   176,    45,    47,   175,
      61,    64,    65,    66,    67,    68,    69,    70,   139,   170,
      29,   182,   183,   184,    52,    53,    54,    71,   162,   112,
      61,    27,    45,    47,    50,   138,   112,   170,    30,   160,
       4,     5,     6,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,   114,   117,   120,
     122,   127,   146,   147,   148,   149,   150,   165,   179,    32,
     122,   171,    33,   123,   173,   183,    72,    73,    74,    76,
      77,   137,   185,   187,   112,   112,   112,   112,   117,   163,
     160,   146,    35,    36,   156,   156,   156,   156,     4,     4,
       4,    13,   123,   150,   151,   165,   115,   124,    13,   148,
     181,    14,    25,    15,    16,    17,    18,    19,    20,    21,
      22,   133,   134,   135,    38,    49,    59,    60,    75,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   129,   130,   131,   132,   191,   192,   194,   195,    30,
      55,    56,   161,     4,    30,    30,   164,   148,   148,   148,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,   133,   134,   136,   148,   153,   118,   118,   113,   123,
     115,    40,   151,   152,     3,     4,     7,     8,     9,    10,
      11,    12,    61,   101,   103,   108,   109,   110,   111,   120,
     129,   130,   131,   132,   154,   178,   179,   180,   180,   180,
     180,   181,    59,   139,   148,   148,   181,   148,   148,   188,
     181,   181,   181,   181,   181,   181,   148,   148,   148,   181,
     139,    99,   100,   113,   119,   113,   144,   144,   144,     8,
       9,     3,     4,   128,     4,     7,    10,    11,    12,   117,
     120,   122,   125,   154,   179,   144,   150,   150,   150,    40,
     148,   167,   168,   169,   113,   116,    63,   177,   115,   115,
     115,   115,   115,   115,   153,   155,   115,   115,   115,   115,
     113,   113,   146,   113,   141,   113,   141,   180,   117,   113,
     113,   113,   113,   113,   113,    39,   180,   180,   180,   113,
     146,   148,   181,    30,    37,    57,   142,   145,   119,   155,
     155,   123,   155,    30,   119,   121,   165,   166,   113,   116,
      40,    30,   153,   153,   153,   153,   153,   153,   113,   121,
     153,   153,   153,   153,    25,    25,   180,    20,    57,    20,
     113,   193,   180,   117,   181,   148,   181,   181,   181,   148,
     113,   113,   113,   181,   180,   180,   113,    30,     4,   144,
     119,   121,   123,    40,   167,   142,   143,   113,   193,   113,
     113,   113,   113,    39,   153,   113,   113,   113,   113,   180,
     180,   115,   180,     4,   180,   181,   189,   113,   180,   113,
     113,   113,   180,   180,   180,   115,   148,    57,   140,    30,
     116,   153,   153,   153,   153,   148,   153,   153,   153,   153,
     113,   117,   189,   190,   113,   141,   141,   113,   180,   113,
     181,   181,   181,   190,   180,     4,   113,   116,   113,   113,
     116,   116,   116,   116,   116,    25,   119,   135,   186,   116,
     181,   119,   180,   116,   153,   153,   153,   180,   178,   119,
     135,    39,   119,   116,   116,   116,   113,   178,    25,    25,
     113,   180,   180,    25,    76,   180,    25,   180
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
        case 39:
#line 207 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " = ";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 40:
#line 211 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string(""); 
  ;}
    break;

  case 48:
#line 218 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 55:
#line 223 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += *(yyvsp[0].Const).cnst; 
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-1].String); 
    ;}
    break;

  case 56:
#line 228 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 57:
#line 233 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 58:
#line 234 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { *(yyvsp[-1].String) += " " + *(yyvsp[0].Const).cnst; delete (yyvsp[0].Const).cnst; (yyval.String) = (yyvsp[-1].String); ;}
    break;

  case 59:
#line 237 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 60:
#line 238 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, ", "); 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Const).cnst;
    delete (yyvsp[0].Const).cnst;
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 61:
#line 246 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 62:
#line 252 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 64:
#line 256 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 65:
#line 257 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      (yyvsp[-1].String)->insert(0, ", ");
      if (!(yyvsp[0].String)->empty())
        *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 67:
#line 267 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      *(yyvsp[-1].String) += " " + *(yyvsp[0].Const).cnst;
      delete (yyvsp[0].Const).cnst;
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 87:
#line 289 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
           (yyval.Type).newTy = (yyvsp[0].String); (yyval.Type).oldTy = OpaqueTy;
         ;}
    break;

  case 88:
#line 295 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Type UpReference
    (yyvsp[0].Const).cnst->insert(0, "\\");
    (yyval.Type).newTy = (yyvsp[0].Const).cnst;
    (yyval.Type).oldTy = OpaqueTy;
  ;}
    break;

  case 89:
#line 300 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {           // Function derived type?
    *(yyvsp[-3].Type).newTy += "( " + *(yyvsp[-1].String) + " )";
    delete (yyvsp[-1].String);
    (yyval.Type).newTy = (yyvsp[-3].Type).newTy;
    (yyval.Type).oldTy = FunctionTy;
  ;}
    break;

  case 90:
#line 306 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Sized array type?
    (yyvsp[-3].Const).cnst->insert(0,"[ ");
    *(yyvsp[-3].Const).cnst += " x " + *(yyvsp[-1].Type).newTy + " ]";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].Const).cnst;
    (yyval.Type).oldTy = ArrayTy;
  ;}
    break;

  case 91:
#line 313 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Packed array type?
    (yyvsp[-3].Const).cnst->insert(0,"< ");
    *(yyvsp[-3].Const).cnst += " x " + *(yyvsp[-1].Type).newTy + " >";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].Const).cnst;
    (yyval.Type).oldTy = PackedTy;
  ;}
    break;

  case 92:
#line 320 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                        // Structure type?
    (yyvsp[-1].String)->insert(0, "{ ");
    *(yyvsp[-1].String) += " }";
    (yyval.Type).newTy = (yyvsp[-1].String);
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 93:
#line 326 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                  // Empty structure type?
    (yyval.Type).newTy = new std::string("{ }");
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 94:
#line 330 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                             // Pointer type?
    *(yyvsp[-1].Type).newTy += '*';
    (yyvsp[-1].Type).oldTy = PointerTy;
    (yyval.Type) = (yyvsp[-1].Type);
  ;}
    break;

  case 95:
#line 340 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].Type).newTy;
  ;}
    break;

  case 96:
#line 343 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Type).newTy;
    delete (yyvsp[0].Type).newTy;
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 98:
#line 352 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 99:
#line 357 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 100:
#line 360 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 101:
#line 370 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " [ " + *(yyvsp[-1].String) + " ]";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 102:
#line 376 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += "[ ]";
  ;}
    break;

  case 103:
#line 381 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " c" + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 104:
#line 387 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " < " + *(yyvsp[-1].String) + " >";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 105:
#line 393 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " { " + *(yyvsp[-1].String) + " }";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 106:
#line 399 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " [ ]";
  ;}
    break;

  case 107:
#line 404 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst +=  " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 108:
#line 410 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 109:
#line 416 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 110:
#line 422 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 111:
#line 428 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 112:
#line 434 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {      // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 113:
#line 440 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {            // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 114:
#line 446 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                      // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 115:
#line 452 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                     // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 116:
#line 458 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Float & Double constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
  ;}
    break;

  case 117:
#line 466 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    // We must infer the cast opcode from the types of the operands. 
    const char *opcode = getCastOpcode((yyvsp[-3].Const).type, (yyvsp[-1].Type));
    (yyval.String) = new std::string(opcode);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + " " + *(yyvsp[-2].String) + " " + *(yyvsp[-1].Type).newTy + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy();
  ;}
    break;

  case 118:
#line 473 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += "(" + *(yyvsp[-2].Const).cnst + " " + *(yyvsp[-1].String) + ")";
    (yyval.String) = (yyvsp[-4].String);
    (yyvsp[-2].Const).destroy();
    delete (yyvsp[-1].String);
  ;}
    break;

  case 119:
#line 479 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 120:
#line 484 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 121:
#line 489 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 122:
#line 494 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 123:
#line 499 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 124:
#line 504 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 125:
#line 509 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 126:
#line 514 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 127:
#line 524 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 128:
#line 529 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Const).cnst); (yyvsp[0].Const).destroy(); ;}
    break;

  case 131:
#line 544 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
;}
    break;

  case 132:
#line 549 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 133:
#line 552 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 134:
#line 557 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "module asm " << " " << *(yyvsp[0].String) << "\n";
    (yyval.String) = 0;
  ;}
    break;

  case 135:
#line 561 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "implementation\n";
    (yyval.String) = 0;
  ;}
    break;

  case 137:
#line 568 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Type).newTy << "\n";
    // delete $2; delete $3; $4.destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 138:
#line 573 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {       // Function prototypes can be in const pool
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 139:
#line 578 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  // Asm blocks can be in the const pool
    *O << *(yyvsp[-2].String) << " " << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 140:
#line 583 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Const).cnst << " " 
       << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Const).destroy(); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 141:
#line 589 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy 
       << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 142:
#line 595 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy 
       << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 143:
#line 601 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy 
       << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 144:
#line 607 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 145:
#line 612 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " = " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 146:
#line 617 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = 0;
  ;}
    break;

  case 150:
#line 627 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 151:
#line 632 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].Const).cnst;
    if (*(yyvsp[0].Const).cnst == "64")
      SizeOfPointer = 64;
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 152:
#line 639 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 153:
#line 644 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 154:
#line 651 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1].String)->insert(0, "[ ");
    *(yyvsp[-1].String) += " ]";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 155:
#line 658 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 157:
#line 664 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 161:
#line 673 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 162:
#line 675 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  (yyval.String) = (yyvsp[-1].Type).newTy;
  if (!(yyvsp[0].String)->empty())
    *(yyval.String) += " " + *(yyvsp[0].String);
  delete (yyvsp[0].String);
;}
    break;

  case 163:
#line 682 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 164:
#line 686 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 165:
#line 690 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 166:
#line 693 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    (yyval.String) = (yyvsp[-2].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 167:
#line 698 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 168:
#line 701 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 169:
#line 704 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-7].String)->empty()) {
      *(yyvsp[-7].String) += " ";
    }
    *(yyvsp[-7].String) += *(yyvsp[-6].Type).newTy + " " + *(yyvsp[-5].String) + "(" + *(yyvsp[-3].String) + ")";
    if (!(yyvsp[-1].String)->empty()) {
      *(yyvsp[-7].String) += " " + *(yyvsp[-1].String);
    }
    if (!(yyvsp[0].String)->empty()) {
      *(yyvsp[-7].String) += " " + *(yyvsp[0].String);
    }
    (yyvsp[-6].Type).destroy();
    delete (yyvsp[-5].String);
    delete (yyvsp[-3].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 170:
#line 723 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string("begin");
  ;}
    break;

  case 171:
#line 726 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = new std::string ("{");
  ;}
    break;

  case 172:
#line 730 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if (!(yyvsp[-2].String)->empty()) {
    *O << *(yyvsp[-2].String) << " ";
  }
  *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
  delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String);
  (yyval.String) = 0;
;}
    break;

  case 173:
#line 739 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("end"); ;}
    break;

  case 174:
#line 740 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); ;}
    break;

  case 175:
#line 742 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if ((yyvsp[-1].String))
    *O << *(yyvsp[-1].String);
  *O << '\n' << *(yyvsp[0].String) << "\n";
  (yyval.String) = 0;
;}
    break;

  case 176:
#line 750 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 179:
#line 756 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[-1].String);
    *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 180:
#line 769 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 182:
#line 773 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 183:
#line 774 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 184:
#line 775 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 185:
#line 776 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 186:
#line 777 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 187:
#line 778 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 188:
#line 779 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 189:
#line 780 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 190:
#line 781 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, "<");
    *(yyvsp[-1].String) += ">";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 192:
#line 787 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty()) {
      *(yyvsp[-4].String) += " " + *(yyvsp[-3].String);
    }
    *(yyvsp[-4].String) += " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 193:
#line 796 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].Const).cnst; ;}
    break;

  case 197:
#line 805 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).type = (yyvsp[-1].Type);
    (yyval.Value).val = new std::string(*(yyvsp[-1].Type).newTy + " ");
    *(yyval.Value).val += *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 198:
#line 812 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  ;}
    break;

  case 199:
#line 814 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Do not allow functions with 0 basic blocks   
  ;}
    break;

  case 200:
#line 821 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) ;
  ;}
    break;

  case 201:
#line 825 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 202:
#line 830 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 203:
#line 833 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 204:
#line 839 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {              // Return with a result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 205:
#line 844 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                       // Return with no result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Type).newTy << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 206:
#line 849 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                         // Unconditional Branch...
    *O << "    " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 207:
#line 854 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].String) << ", " 
       << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].String) << ", " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); delete (yyvsp[-6].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); 
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 208:
#line 861 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].String) << ", " << *(yyvsp[-4].Type).newTy 
       << " " << *(yyvsp[-3].String) << " [" << *(yyvsp[-1].String) << " ]\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); delete (yyvsp[-6].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 209:
#line 867 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-7].String) << " " << *(yyvsp[-6].Type).newTy << " " << *(yyvsp[-5].String) << ", " 
       << *(yyvsp[-3].Type).newTy << " " << *(yyvsp[-2].String) << "[]\n";
    delete (yyvsp[-7].String); (yyvsp[-6].Type).destroy(); delete (yyvsp[-5].String); (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String);
    (yyval.String) = 0;
  ;}
    break;

  case 210:
#line 874 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    ";
    if (!(yyvsp[-13].String)->empty())
      *O << *(yyvsp[-13].String);
    *O << *(yyvsp[-12].String) << " " << *(yyvsp[-11].String) << " " << *(yyvsp[-10].Type).newTy << " " << *(yyvsp[-9].String) << " ("
       << *(yyvsp[-7].String) << ") " << *(yyvsp[-5].String) << " " << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].String) << " " 
       << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-13].String); delete (yyvsp[-12].String); delete (yyvsp[-11].String); (yyvsp[-10].Type).destroy(); delete (yyvsp[-9].String); delete (yyvsp[-7].String); 
    delete (yyvsp[-5].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); 
    delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 211:
#line 886 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 212:
#line 891 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 213:
#line 897 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].String) + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 214:
#line 902 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-3].String)->insert(0, *(yyvsp[-4].Type).newTy + " " );
    *(yyvsp[-3].String) += ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 215:
#line 910 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
  ;}
    break;

  case 216:
#line 917 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {    // Used for PHI nodes
    (yyvsp[-3].String)->insert(0, *(yyvsp[-5].Type).newTy + "[");
    *(yyvsp[-3].String) += "," + *(yyvsp[-1].String) + "]";
    (yyvsp[-5].Type).destroy(); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 217:
#line 923 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6].String) += ", [" + *(yyvsp[-3].String) + "," + *(yyvsp[-1].String) + "]";
    delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 218:
#line 931 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Value).val); (yyvsp[0].Value).destroy(); ;}
    break;

  case 219:
#line 932 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 221:
#line 941 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 222:
#line 945 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 224:
#line 953 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 225:
#line 958 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 226:
#line 963 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 227:
#line 968 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 228:
#line 973 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 229:
#line 978 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char *opcode = getCastOpcode((yyvsp[-2].Value).type, (yyvsp[0].Type));
    (yyval.String) = new std::string(opcode);
    *(yyval.String) += *(yyvsp[-2].Value).val + " " + *(yyvsp[-1].String) + " " + *(yyvsp[0].Type).newTy; 
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy();
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
  ;}
    break;

  case 230:
#line 985 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 231:
#line 990 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Type).newTy;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Type).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 232:
#line 995 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 233:
#line 1000 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 234:
#line 1005 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 235:
#line 1010 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 236:
#line 1015 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-6].String) += " " + *(yyvsp[-5].String);
    if (!(yyvsp[-6].String)->empty())
      *(yyvsp[-6].String) += " ";
    *(yyvsp[-6].String) += *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].String) + "(" + *(yyvsp[-1].String) + ")";
    delete (yyvsp[-5].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 238:
#line 1029 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[0].String)->insert(0, ", ");
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 239:
#line 1033 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  (yyval.String) = new std::string(); ;}
    break;

  case 241:
#line 1038 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 242:
#line 1041 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 243:
#line 1048 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 244:
#line 1055 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 245:
#line 1062 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 246:
#line 1069 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 247:
#line 1074 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty())
      *(yyvsp[-3].String) += " ";
    *(yyvsp[-3].String) += *(yyvsp[-2].String) + " " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 248:
#line 1081 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-5].String) += " ";
    *(yyvsp[-5].String) += *(yyvsp[-4].String) + " " + *(yyvsp[-3].Value).val + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    delete (yyvsp[-4].String); (yyvsp[-3].Value).destroy(); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 249:
#line 1088 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String) + " " + *(yyvsp[0].String);
    (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 3373 "UpgradeParser.tab.c"

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


#line 1094 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"


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

