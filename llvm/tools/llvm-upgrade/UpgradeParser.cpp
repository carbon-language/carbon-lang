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
     SHR = 360,
     ASHR = 361,
     LSHR = 362,
     VAARG = 363,
     EXTRACTELEMENT = 364,
     INSERTELEMENT = 365,
     SHUFFLEVECTOR = 366,
     CAST = 367
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
#define SHR 360
#define ASHR 361
#define LSHR 362
#define VAARG 363
#define EXTRACTELEMENT 364
#define INSERTELEMENT 365
#define SHUFFLEVECTOR 366
#define CAST 367




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
#line 438 "UpgradeParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 450 "UpgradeParser.tab.c"

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
#define YYNTOKENS  127
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  70
/* YYNRULES -- Number of rules. */
#define YYNRULES  250
/* YYNRULES -- Number of states. */
#define YYNSTATES  509

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   367

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     116,   117,   125,     2,   114,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     121,   113,   122,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   118,   115,   120,     2,     2,     2,     2,     2,   126,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     119,     2,     2,   123,     2,   124,     2,     2,     2,     2,
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
     105,   106,   107,   108,   109,   110,   111,   112
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
     182,   187,   193,   199,   203,   206,   209,   211,   215,   217,
     221,   223,   224,   229,   233,   237,   242,   247,   251,   254,
     257,   260,   263,   266,   269,   272,   275,   278,   281,   288,
     294,   303,   310,   317,   324,   331,   338,   347,   356,   360,
     362,   364,   366,   368,   371,   374,   379,   382,   384,   389,
     392,   397,   404,   411,   418,   425,   429,   434,   435,   437,
     439,   441,   445,   449,   453,   457,   461,   465,   467,   468,
     470,   472,   474,   475,   478,   482,   484,   486,   490,   492,
     493,   502,   504,   506,   510,   512,   514,   518,   519,   521,
     523,   527,   528,   530,   532,   534,   536,   538,   540,   542,
     544,   546,   550,   552,   558,   560,   562,   564,   566,   569,
     572,   574,   577,   580,   581,   583,   586,   589,   593,   603,
     613,   622,   637,   639,   641,   648,   654,   657,   664,   672,
     674,   678,   680,   681,   684,   686,   692,   698,   704,   707,
     712,   717,   724,   729,   734,   741,   748,   751,   759,   761,
     764,   765,   767,   768,   772,   779,   783,   790,   793,   798,
     805
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     158,     0,    -1,    19,    -1,    20,    -1,    17,    -1,    18,
      -1,    78,    -1,    79,    -1,    80,    -1,    81,    -1,    82,
      -1,    83,    -1,    84,    -1,    85,    -1,    86,    -1,    87,
      -1,    88,    -1,    89,    -1,    90,    -1,    91,    -1,    92,
      -1,    93,    -1,    94,    -1,    95,    -1,   104,    -1,   105,
      -1,   106,    -1,   107,    -1,    11,    -1,     9,    -1,     7,
      -1,     5,    -1,    12,    -1,    10,    -1,     8,    -1,     6,
      -1,   134,    -1,   135,    -1,    13,    -1,    14,    -1,   166,
     113,    -1,    -1,    42,    -1,    43,    -1,    44,    -1,    48,
      -1,    45,    -1,    46,    -1,    47,    -1,    -1,    65,    -1,
      66,    -1,    67,    -1,    68,    -1,    69,    -1,    70,    -1,
      64,    18,    -1,    -1,    -1,    57,    18,    -1,    -1,   114,
      57,    18,    -1,    37,    30,    -1,    -1,   143,    -1,    -1,
     114,   146,   145,    -1,   143,    -1,    57,    18,    -1,   149,
      -1,     3,    -1,   151,    -1,     3,    -1,   151,    -1,     4,
      -1,     5,    -1,     6,    -1,     7,    -1,     8,    -1,     9,
      -1,    10,    -1,    11,    -1,    12,    -1,    13,    -1,    14,
      -1,    15,    -1,    16,    -1,   150,    -1,   180,    -1,   115,
      18,    -1,   148,   116,   153,   117,    -1,   118,    18,   119,
     151,   120,    -1,   121,    18,   119,   151,   122,    -1,   123,
     152,   124,    -1,   123,   124,    -1,   151,   125,    -1,   151,
      -1,   152,   114,   151,    -1,   152,    -1,   152,   114,    40,
      -1,    40,    -1,    -1,   149,   118,   156,   120,    -1,   149,
     118,   120,    -1,   149,   126,    30,    -1,   149,   121,   156,
     122,    -1,   149,   123,   156,   124,    -1,   149,   123,   124,
      -1,   149,    22,    -1,   149,    23,    -1,   149,   180,    -1,
     149,   155,    -1,   149,    24,    -1,   134,   129,    -1,   135,
      18,    -1,     4,    25,    -1,     4,    26,    -1,   137,    21,
      -1,   112,   116,   154,    39,   149,   117,    -1,   101,   116,
     154,   194,   117,    -1,   103,   116,   154,   114,   154,   114,
     154,   117,    -1,   130,   116,   154,   114,   154,   117,    -1,
     131,   116,   154,   114,   154,   117,    -1,   132,   116,   154,
     114,   154,   117,    -1,   133,   116,   154,   114,   154,   117,
      -1,   109,   116,   154,   114,   154,   117,    -1,   110,   116,
     154,   114,   154,   114,   154,   117,    -1,   111,   116,   154,
     114,   154,   114,   154,   117,    -1,   156,   114,   154,    -1,
     154,    -1,    35,    -1,    36,    -1,   159,    -1,   159,   175,
      -1,   159,   177,    -1,   159,    62,    61,   161,    -1,   159,
      31,    -1,   160,    -1,   160,   138,    27,   147,    -1,   160,
     177,    -1,   160,    62,    61,   161,    -1,   160,   138,   139,
     157,   154,   145,    -1,   160,   138,    50,   157,   149,   145,
      -1,   160,   138,    45,   157,   149,   145,    -1,   160,   138,
      47,   157,   149,   145,    -1,   160,    51,   163,    -1,   160,
      58,   113,   164,    -1,    -1,    30,    -1,    56,    -1,    55,
      -1,    53,   113,   162,    -1,    54,   113,    18,    -1,    52,
     113,    30,    -1,    71,   113,    30,    -1,   118,   165,   120,
      -1,   165,   114,    30,    -1,    30,    -1,    -1,    28,    -1,
      30,    -1,   166,    -1,    -1,   149,   167,    -1,   169,   114,
     168,    -1,   168,    -1,   169,    -1,   169,   114,    40,    -1,
      40,    -1,    -1,   140,   147,   166,   116,   170,   117,   144,
     141,    -1,    32,    -1,   123,    -1,   139,   171,   172,    -1,
      33,    -1,   124,    -1,   173,   183,   174,    -1,    -1,    45,
      -1,    47,    -1,    34,   176,   171,    -1,    -1,    63,    -1,
      17,    -1,    18,    -1,    21,    -1,    25,    -1,    26,    -1,
      22,    -1,    23,    -1,    24,    -1,   121,   156,   122,    -1,
     155,    -1,    61,   178,    30,   114,    30,    -1,   128,    -1,
     166,    -1,   180,    -1,   179,    -1,   149,   181,    -1,   183,
     184,    -1,   184,    -1,   185,   186,    -1,   185,   188,    -1,
      -1,    29,    -1,    72,   182,    -1,    72,     3,    -1,    73,
      15,   181,    -1,    73,     4,   181,   114,    15,   181,   114,
      15,   181,    -1,    74,   136,   181,   114,    15,   181,   118,
     187,   120,    -1,    74,   136,   181,   114,    15,   181,   118,
     120,    -1,   138,    75,   140,   147,   181,   116,   191,   117,
      39,    15,   181,    76,    15,   181,    -1,    76,    -1,    77,
      -1,   187,   136,   179,   114,    15,   181,    -1,   136,   179,
     114,    15,   181,    -1,   138,   193,    -1,   149,   118,   181,
     114,   181,   120,    -1,   189,   114,   118,   181,   114,   181,
     120,    -1,   182,    -1,   190,   114,   182,    -1,   190,    -1,
      -1,    60,    59,    -1,    59,    -1,   130,   149,   181,   114,
     181,    -1,   131,   149,   181,   114,   181,    -1,   132,   149,
     181,   114,   181,    -1,    49,   182,    -1,   133,   182,   114,
     182,    -1,   112,   182,    39,   149,    -1,   103,   182,   114,
     182,   114,   182,    -1,   108,   182,   114,   149,    -1,   109,
     182,   114,   182,    -1,   110,   182,   114,   182,   114,   182,
      -1,   111,   182,   114,   182,   114,   182,    -1,   102,   189,
      -1,   192,   140,   147,   181,   116,   191,   117,    -1,   196,
      -1,   114,   190,    -1,    -1,    38,    -1,    -1,    96,   149,
     142,    -1,    96,   149,   114,    10,   181,   142,    -1,    97,
     149,   142,    -1,    97,   149,   114,    10,   181,   142,    -1,
      98,   182,    -1,   195,    99,   149,   181,    -1,   195,   100,
     182,   114,   149,   181,    -1,   101,   149,   181,   194,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   188,   188,   188,   189,   189,   193,   193,   193,   193,
     193,   193,   193,   193,   193,   194,   194,   194,   195,   195,
     195,   195,   195,   195,   196,   196,   196,   196,   200,   200,
     200,   200,   201,   201,   201,   201,   202,   202,   203,   203,
     206,   210,   215,   215,   215,   215,   215,   215,   216,   217,
     220,   220,   220,   220,   220,   221,   222,   227,   232,   233,
     236,   237,   245,   251,   252,   255,   256,   265,   266,   279,
     279,   280,   280,   281,   285,   285,   285,   285,   285,   285,
     285,   286,   286,   286,   286,   286,   287,   287,   288,   294,
     299,   305,   312,   319,   325,   329,   339,   342,   350,   351,
     356,   359,   369,   375,   380,   386,   392,   398,   403,   409,
     415,   421,   427,   433,   439,   445,   451,   457,   465,   472,
     478,   483,   488,   493,   498,   506,   511,   516,   526,   531,
     536,   536,   546,   551,   554,   559,   563,   567,   570,   575,
     580,   585,   591,   597,   603,   609,   614,   619,   624,   626,
     626,   629,   634,   641,   646,   653,   660,   665,   666,   674,
     674,   675,   675,   677,   684,   688,   692,   695,   700,   703,
     705,   725,   728,   732,   741,   742,   744,   752,   753,   754,
     758,   771,   772,   775,   775,   775,   775,   775,   775,   775,
     776,   777,   782,   783,   792,   792,   795,   795,   801,   808,
     811,   819,   823,   828,   831,   837,   842,   847,   852,   859,
     865,   871,   884,   889,   895,   900,   908,   915,   921,   929,
     930,   938,   939,   943,   948,   951,   956,   961,   966,   971,
     979,   986,   991,   996,  1001,  1006,  1011,  1016,  1025,  1030,
    1034,  1038,  1039,  1042,  1049,  1056,  1063,  1070,  1075,  1082,
    1089
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "VOID", "BOOL", "SBYTE", "UBYTE",
  "SHORT", "USHORT", "INT", "UINT", "LONG", "ULONG", "FLOAT", "DOUBLE",
  "LABEL", "OPAQUE", "ESINT64VAL", "EUINT64VAL", "SINTVAL", "UINTVAL",
  "FPVAL", "NULL_TOK", "UNDEF", "ZEROINITIALIZER", "TRUETOK", "FALSETOK",
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
  "LOAD", "STORE", "GETELEMENTPTR", "PHI_TOK", "SELECT", "SHL", "SHR",
  "ASHR", "LSHR", "VAARG", "EXTRACTELEMENT", "INSERTELEMENT",
  "SHUFFLEVECTOR", "CAST", "'='", "','", "'\\\\'", "'('", "')'", "'['",
  "'x'", "']'", "'<'", "'>'", "'{'", "'}'", "'*'", "'c'", "$accept",
  "IntVal", "EInt64Val", "ArithmeticOps", "LogicalOps", "SetCondOps",
  "ShiftOps", "SIntType", "UIntType", "IntType", "FPType", "OptAssign",
  "OptLinkage", "OptCallingConv", "OptAlign", "OptCAlign", "SectionString",
  "OptSection", "GlobalVarAttributes", "GlobalVarAttribute", "TypesV",
  "UpRTypesV", "Types", "PrimType", "UpRTypes", "TypeListI",
  "ArgTypeListI", "ConstVal", "ConstExpr", "ConstVector", "GlobalType",
  "Module", "DefinitionList", "ConstPool", "AsmBlock", "BigOrLittle",
  "TargetDefinition", "LibrariesDefinition", "LibList", "Name", "OptName",
  "ArgVal", "ArgListH", "ArgList", "FunctionHeaderH", "BEGIN",
  "FunctionHeader", "END", "Function", "FnDeclareLinkage", "FunctionProto",
  "OptSideEffect", "ConstValueRef", "SymbolicValueRef", "ValueRef",
  "ResolvedVal", "BasicBlockList", "BasicBlock", "InstructionList",
  "BBTerminatorInst", "JumpTable", "Inst", "PHIList", "ValueRefList",
  "ValueRefListE", "OptTailCall", "InstVal", "IndexList", "OptVolatile",
  "MemoryInst", 0
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
     365,   366,   367,    61,    44,    92,    40,    41,    91,   120,
      93,    60,    62,   123,   125,    42,    99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   127,   128,   128,   129,   129,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   131,   131,   131,   132,   132,
     132,   132,   132,   132,   133,   133,   133,   133,   134,   134,
     134,   134,   135,   135,   135,   135,   136,   136,   137,   137,
     138,   138,   139,   139,   139,   139,   139,   139,   139,   139,
     140,   140,   140,   140,   140,   140,   140,   140,   141,   141,
     142,   142,   143,   144,   144,   145,   145,   146,   146,   147,
     147,   148,   148,   149,   150,   150,   150,   150,   150,   150,
     150,   150,   150,   150,   150,   150,   151,   151,   151,   151,
     151,   151,   151,   151,   151,   151,   152,   152,   153,   153,
     153,   153,   154,   154,   154,   154,   154,   154,   154,   154,
     154,   154,   154,   154,   154,   154,   154,   154,   155,   155,
     155,   155,   155,   155,   155,   155,   155,   155,   156,   156,
     157,   157,   158,   159,   159,   159,   159,   159,   160,   160,
     160,   160,   160,   160,   160,   160,   160,   160,   161,   162,
     162,   163,   163,   163,   163,   164,   165,   165,   165,   166,
     166,   167,   167,   168,   169,   169,   170,   170,   170,   170,
     171,   172,   172,   173,   174,   174,   175,   176,   176,   176,
     177,   178,   178,   179,   179,   179,   179,   179,   179,   179,
     179,   179,   179,   179,   180,   180,   181,   181,   182,   183,
     183,   184,   185,   185,   185,   186,   186,   186,   186,   186,
     186,   186,   186,   186,   187,   187,   188,   189,   189,   190,
     190,   191,   191,   192,   192,   193,   193,   193,   193,   193,
     193,   193,   193,   193,   193,   193,   193,   193,   193,   194,
     194,   195,   195,   196,   196,   196,   196,   196,   196,   196,
     196
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
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       4,     5,     5,     3,     2,     2,     1,     3,     1,     3,
       1,     0,     4,     3,     3,     4,     4,     3,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     6,     5,
       8,     6,     6,     6,     6,     6,     8,     8,     3,     1,
       1,     1,     1,     2,     2,     4,     2,     1,     4,     2,
       4,     6,     6,     6,     6,     3,     4,     0,     1,     1,
       1,     3,     3,     3,     3,     3,     3,     1,     0,     1,
       1,     1,     0,     2,     3,     1,     1,     3,     1,     0,
       8,     1,     1,     3,     1,     1,     3,     0,     1,     1,
       3,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     5,     1,     1,     1,     1,     2,     2,
       1,     2,     2,     0,     1,     2,     2,     3,     9,     9,
       8,    14,     1,     1,     6,     5,     2,     6,     7,     1,
       3,     1,     0,     2,     1,     5,     5,     5,     2,     4,
       4,     6,     4,     4,     6,     6,     2,     7,     1,     2,
       0,     1,     0,     3,     6,     3,     6,     2,     4,     6,
       4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
     147,     0,    49,   137,     1,   136,   177,    42,    43,    44,
      46,    47,    48,    45,     0,    57,   203,   133,   134,   159,
     160,     0,     0,     0,    49,     0,   139,   178,   179,    57,
       0,     0,    50,    51,    52,    53,    54,    55,     0,     0,
     204,   203,   200,    41,     0,     0,     0,     0,   145,     0,
       0,     0,     0,     0,     0,     0,    40,   180,   148,   135,
      56,    70,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,     2,     3,     0,     0,     0,
       0,   194,     0,     0,    69,    87,    73,   195,    88,   171,
     172,   173,   174,   175,   176,   199,     0,     0,     0,   212,
     213,   242,   201,   202,     0,     0,     0,     0,   158,   146,
     140,   138,   130,   131,     0,     0,     0,     0,    89,     0,
       0,    72,    94,    96,     0,     0,   101,    95,   206,     0,
     205,     0,     0,    31,    35,    30,    34,    29,    33,    28,
      32,    36,    37,     0,   241,     0,   224,     0,    57,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,     0,     0,     0,
       0,     0,     0,    24,    25,    26,    27,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    57,   216,     0,   238,
     153,   150,   149,   151,   152,   154,   157,     0,    65,    65,
      65,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,     0,     0,     0,     0,    65,     0,     0,     0,
      93,   169,   100,    98,     0,   183,   184,   185,   188,   189,
     190,   186,   187,   181,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   192,   197,   196,   198,     0,
     207,     0,   228,   223,     0,    60,    60,   247,     0,     0,
     236,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   155,     0,   143,   144,   142,
     115,   116,     4,     5,   113,   114,   117,   108,   109,   112,
       0,     0,     0,     0,   111,   110,   141,    71,    71,    97,
     168,   162,   165,   166,     0,     0,    90,   182,     0,     0,
       0,     0,     0,     0,     0,   129,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   243,     0,   245,   240,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   156,     0,     0,    67,    65,   103,
       0,     0,   107,     0,   104,    91,    92,   161,   163,     0,
      63,    99,     0,   240,     0,     0,     0,     0,     0,     0,
     191,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   250,     0,     0,     0,   232,   233,     0,     0,
     230,     0,     0,     0,   229,     0,   248,     0,    62,    68,
      66,   102,   105,   106,   167,   164,    64,    58,     0,     0,
       0,     0,     0,     0,     0,   128,     0,     0,     0,     0,
       0,     0,   222,    60,    61,    60,   219,   239,     0,     0,
       0,     0,     0,   225,   226,   227,   222,     0,     0,   170,
     193,   119,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   221,     0,     0,   244,   246,     0,     0,
       0,   231,   234,   235,     0,   249,    59,     0,   125,     0,
       0,   118,   121,   122,   123,   124,     0,   210,     0,     0,
       0,   220,   217,     0,   237,     0,     0,     0,   208,     0,
     209,     0,     0,   218,   120,   126,   127,     0,     0,     0,
       0,     0,     0,   215,     0,     0,   214,     0,   211
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,    81,   284,   241,   242,   243,   244,   212,   213,   143,
     214,    24,    15,    38,   439,   325,   347,   407,   277,   348,
      82,    83,   215,    85,    86,   124,   224,   315,   245,   316,
     114,     1,     2,     3,    59,   193,    48,   109,   197,    87,
     358,   302,   303,   304,    39,    91,    16,    94,    17,    29,
      18,   308,   246,    88,   248,   426,    41,    42,    43,   102,
     479,   103,   260,   453,   454,   186,   187,   382,   188,   189
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -462
static const short int yypact[] =
{
    -462,    20,   223,   382,  -462,  -462,    74,  -462,  -462,  -462,
    -462,  -462,  -462,  -462,   -37,   262,    13,  -462,  -462,  -462,
    -462,    64,   -82,     9,    -9,   -15,  -462,  -462,  -462,   262,
      79,   110,  -462,  -462,  -462,  -462,  -462,  -462,   688,   -17,
    -462,   -19,  -462,    -5,    38,    40,    50,    59,  -462,    57,
      79,   688,   132,   132,   132,   132,  -462,  -462,  -462,  -462,
    -462,    66,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,   173,   184,   193,
     469,  -462,   108,   103,  -462,  -462,    -3,  -462,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,   717,    17,   235,  -462,
    -462,  1038,  -462,  -462,   191,   114,   204,   194,   195,  -462,
    -462,  -462,  -462,  -462,   745,   745,   745,   774,  -462,   109,
     111,  -462,  -462,    -3,   -13,   113,    80,  -462,    66,   932,
    -462,   932,   932,  -462,  -462,  -462,  -462,  -462,  -462,  -462,
    -462,  -462,  -462,   932,  -462,   745,  -462,   178,   262,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,   745,   745,   745,
     745,   745,   745,  -462,  -462,  -462,  -462,   745,   745,   745,
     745,   745,   745,   745,   745,   745,   262,  -462,    78,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,   -40,   117,   117,
     117,   155,   172,   214,   176,   230,   179,   231,   182,   232,
     234,   237,   200,   233,   238,   822,   117,   745,   745,   745,
    -462,   527,  -462,   139,   143,  -462,  -462,  -462,  -462,  -462,
    -462,  -462,  -462,   211,   140,   160,   162,   163,   166,   167,
     774,   168,   170,   171,   175,  -462,  -462,  -462,  -462,   185,
    -462,   186,  -462,  -462,   688,   189,   190,  -462,   932,   174,
     196,   201,   202,   205,   206,   208,   270,   932,   932,   932,
     210,   688,   745,   745,   288,  -462,    16,  -462,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,
     566,   774,   499,   303,  -462,  -462,  -462,   -23,     2,    -3,
    -462,   108,  -462,   225,   220,   596,  -462,  -462,   314,   774,
     774,   774,   774,   774,   774,  -462,   -95,   774,   774,   774,
     774,   330,   331,   932,    -2,  -462,    -1,  -462,   236,   932,
     240,   745,   745,   745,   745,   745,   745,   239,   246,   247,
     745,   932,   932,   249,  -462,   318,   333,  -462,   117,  -462,
     -39,     1,  -462,   -10,  -462,  -462,  -462,  -462,  -462,   648,
     312,  -462,   252,   236,   253,   254,   257,   261,   313,   774,
    -462,   263,   265,   266,   267,   932,   932,   272,   932,   336,
     932,   745,  -462,   275,   932,   276,  -462,  -462,   277,   281,
    -462,   932,   932,   932,  -462,   283,  -462,   745,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,   343,   372,   286,
     774,   774,   774,   774,   745,  -462,   774,   774,   774,   774,
     291,   289,   745,   292,  -462,   292,  -462,   294,   932,   297,
     745,   745,   745,  -462,  -462,  -462,   745,   932,   395,  -462,
    -462,  -462,   300,   298,   305,   307,   306,   317,   319,   324,
     325,   420,    39,   294,   326,   388,  -462,  -462,   745,   327,
     932,  -462,  -462,  -462,   329,  -462,  -462,   774,  -462,   774,
     774,  -462,  -462,  -462,  -462,  -462,   932,  -462,   979,    54,
     409,  -462,  -462,   332,  -462,   334,   338,   339,  -462,   335,
    -462,   979,   435,  -462,  -462,  -462,  -462,   438,   344,   932,
     932,   442,   383,  -462,   932,   445,  -462,   932,  -462
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -462,  -462,  -462,   360,   361,   362,   363,   -97,   -96,  -400,
    -462,   423,   443,  -132,  -462,  -252,   126,  -462,  -194,  -462,
     -44,  -462,   -38,  -462,   -69,   342,  -462,  -105,   256,  -134,
     107,  -462,  -462,  -462,   419,  -462,  -462,  -462,  -462,     0,
    -462,   128,  -462,  -462,   461,  -462,  -462,  -462,  -462,  -462,
     488,  -462,  -461,  -103,    -6,     7,  -462,   451,  -462,  -462,
    -462,  -462,  -462,   115,    81,  -462,  -462,   137,  -462,  -462
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -133
static const short int yytable[] =
{
      84,   141,   142,    25,   327,   278,   279,   111,   378,   380,
      40,   123,   216,    84,    92,    89,   254,   489,    51,   369,
       4,   131,   296,    19,    30,    20,   247,   370,   247,   247,
     498,    49,   132,     7,     8,     9,    52,    11,    53,    13,
     247,    54,    40,    25,   133,   134,   135,   136,   137,   138,
     139,   140,   478,   345,   271,   379,   379,   123,   129,   133,
     134,   135,   136,   137,   138,   139,   140,    96,    97,    98,
      50,    99,   100,   346,   274,   369,   198,   199,   200,   491,
     275,   401,   125,   121,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,   355,    56,    75,
      76,   219,   127,   130,   369,    93,    90,   129,    19,    58,
      20,   220,   295,   -71,   403,   369,    44,    45,    46,    27,
     222,    28,   127,   402,   356,   249,   250,   127,    60,   255,
     256,   129,   258,   259,   129,    47,    19,   251,    20,   129,
     129,   129,   129,   129,   267,   268,   269,   129,   297,   298,
     299,   104,   252,   105,   400,   247,   350,   351,   353,   477,
     115,   116,   117,   106,   247,   247,   247,   112,   113,   191,
     192,   456,   107,   457,   490,   108,   257,   272,   273,   261,
     280,   281,   -72,   301,   262,   263,   264,   265,   266,   -31,
     -31,   118,   270,   -30,   -30,    77,   -29,   -29,    78,   -28,
     -28,    79,   119,    80,   363,   364,   365,   366,   367,   368,
     323,   120,   371,   372,   373,   374,    84,   282,   283,   126,
     247,   190,   194,  -132,   195,   196,   247,   341,   217,   221,
     218,   276,   -35,    84,   342,   129,   299,   253,   247,   247,
     133,   134,   135,   136,   137,   138,   139,   140,   -34,   -33,
     -32,   285,   328,   305,     5,   -38,   309,     6,   -39,   286,
     306,   337,   338,   339,   415,     7,     8,     9,    10,    11,
      12,    13,   247,   247,   307,   247,   310,   247,   311,   312,
     343,   247,   313,   314,   317,    14,   318,   319,   247,   247,
     247,   320,   329,   129,   386,   129,   129,   129,   390,   321,
     322,   357,   129,   324,   326,   442,   443,   444,   445,   336,
     330,   447,   448,   449,   450,   331,   332,   377,   344,   333,
     334,   301,   335,   383,   340,   247,    31,    32,    33,    34,
      35,    36,    37,   354,   247,   395,   396,   360,   385,   359,
     387,   388,   389,   129,   362,   375,   376,   394,   398,   345,
     381,   399,   414,   391,   424,   141,   142,   247,   384,   437,
     392,   393,   485,   397,   486,   487,   408,   410,   411,   420,
     421,   412,   423,   247,   425,   413,   446,   416,   429,   417,
     418,   419,   141,   142,   129,   433,   434,   435,   422,   428,
     430,   431,   129,   129,   129,   432,   247,   247,   129,   436,
     438,   247,   440,   441,   247,   451,   455,   452,   458,   -41,
      19,   460,    20,   466,   467,   468,     6,   -41,   -41,   469,
     129,   470,   459,   471,   -41,   -41,   -41,   -41,   -41,   -41,
     -41,   465,   -41,    21,   472,   476,   473,   461,   462,   463,
      22,   474,   475,   480,    23,   379,   484,   482,   492,   497,
     499,   494,   493,   500,   483,   495,   496,   504,   501,   505,
     507,   182,   183,   184,   185,   481,   101,    55,   223,   110,
     488,   294,   121,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,   406,   405,    75,    76,
      57,    26,    95,   502,   503,     0,   427,    19,   506,    20,
     409,   508,   121,   201,   202,   203,   204,   205,   206,   207,
     208,   209,   210,   211,    73,    74,     0,   464,    75,    76,
       0,     0,     0,     0,     0,     0,     0,    19,     0,    20,
     121,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,     0,     0,    75,    76,     0,     0,
       0,     0,     0,     0,     0,    19,     0,    20,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   300,     0,   121,
     201,   202,   203,   204,   205,   206,   207,   208,   209,   210,
     211,    73,    74,     0,    77,    75,    76,    78,     0,     0,
      79,     0,    80,   122,    19,     0,    20,     0,     0,   121,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,     0,    77,    75,    76,    78,     0,     0,
      79,     0,    80,   352,    19,     0,    20,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   361,     0,     0,     0,
       0,     0,    77,     0,     0,    78,     0,     0,    79,     0,
      80,   121,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,     0,     0,    75,    76,     0,
       0,     0,     0,     0,     0,     0,    19,     0,    20,     0,
       0,    77,     0,     0,    78,     0,   349,    79,   404,    80,
       0,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,     0,     0,    75,    76,     0,
       0,    77,     0,     0,    78,     0,    19,    79,    20,    80,
     128,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,     0,     0,    75,    76,     0,     0,
       0,     0,     0,     0,     0,    19,     0,    20,   121,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,     0,    77,    75,    76,    78,     0,     0,    79,
       0,    80,     0,    19,     0,    20,     0,   121,   201,   202,
     203,   204,   205,   206,   207,   208,   209,   210,   211,    73,
      74,     0,     0,    75,    76,     0,     0,     0,     0,     0,
       0,     0,    19,    77,    20,     0,    78,     0,     0,    79,
       0,    80,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    77,     0,     0,    78,     0,     0,    79,     0,
      80,    75,    76,     0,   287,   288,   289,     0,     0,     0,
      19,     0,    20,     0,     0,     0,     0,     0,     0,     0,
      77,     0,     0,    78,     0,     0,    79,     0,    80,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    77,
       0,     0,    78,     0,     0,    79,     0,    80,     0,     0,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,     0,     0,
       0,     0,     0,   234,     0,   235,   173,   174,   175,   176,
       0,   236,   237,   238,   239,     0,     0,     0,     0,     0,
     290,     0,     0,   291,     0,   292,     0,     0,   293,   225,
     226,    75,    76,   227,   228,   229,   230,   231,   232,     0,
      19,     0,    20,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   233,     0,     0,   225,   226,     0,     0,
     227,   228,   229,   230,   231,   232,     0,     0,     0,     0,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,     0,     0,
       0,     0,     0,   234,     0,   235,   173,   174,   175,   176,
     233,   236,   237,   238,   239,     0,     0,     0,     0,     0,
       0,     0,     0,   240,     0,     0,     0,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,     0,   144,     0,     0,     0,
     234,     0,   235,   173,   174,   175,   176,   145,   236,   237,
     238,   239,     0,     0,     0,     0,     0,   146,   147,     0,
     240,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   148,     0,     0,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,     0,     0,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181
};

static const short int yycheck[] =
{
      38,    98,    98,     3,   256,   199,   200,    51,    10,    10,
      29,    80,   117,    51,    33,    32,   148,   478,    27,   114,
       0,     4,   216,    28,    61,    30,   129,   122,   131,   132,
     491,   113,    15,    42,    43,    44,    45,    46,    47,    48,
     143,    50,    29,    43,     5,     6,     7,     8,     9,    10,
      11,    12,   452,    37,   186,    57,    57,   126,    96,     5,
       6,     7,     8,     9,    10,    11,    12,    72,    73,    74,
      61,    76,    77,    57,   114,   114,   114,   115,   116,   479,
     120,   120,    82,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,   120,   113,    19,
      20,   114,   125,    96,   114,   124,   123,   145,    28,    30,
      30,   124,   215,   116,   124,   114,    52,    53,    54,    45,
      40,    47,   125,   122,   122,   131,   132,   125,    18,   167,
     168,   169,   170,   171,   172,    71,    28,   143,    30,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   217,   218,
     219,   113,   145,   113,   348,   258,   290,   291,   292,   120,
      53,    54,    55,   113,   267,   268,   269,    35,    36,    55,
      56,   423,   113,   425,   120,   118,   169,    99,   100,   172,
      25,    26,   116,   221,   177,   178,   179,   180,   181,    17,
      18,    18,   185,    17,    18,   115,    17,    18,   118,    17,
      18,   121,    18,   123,   309,   310,   311,   312,   313,   314,
     254,    18,   317,   318,   319,   320,   254,    17,    18,   116,
     323,    30,    18,     0,    30,    30,   329,   271,   119,   116,
     119,   114,    18,   271,   272,   273,   305,    59,   341,   342,
       5,     6,     7,     8,     9,    10,    11,    12,    18,    18,
      18,    18,   258,   114,    31,    21,   116,    34,    21,    21,
     117,   267,   268,   269,   369,    42,    43,    44,    45,    46,
      47,    48,   375,   376,    63,   378,   116,   380,   116,   116,
     273,   384,   116,   116,   116,    62,   116,   116,   391,   392,
     393,   116,   118,   331,   332,   333,   334,   335,   336,   114,
     114,   301,   340,   114,   114,   410,   411,   412,   413,    39,
     114,   416,   417,   418,   419,   114,   114,   323,    30,   114,
     114,   359,   114,   329,   114,   428,    64,    65,    66,    67,
      68,    69,    70,    30,   437,   341,   342,   117,   331,   114,
     333,   334,   335,   381,    30,    15,    15,   340,    30,    37,
     114,    18,    39,   114,    18,   452,   452,   460,   118,   397,
     114,   114,   467,   114,   469,   470,   114,   114,   114,   375,
     376,   114,   378,   476,   380,   114,   414,   114,   384,   114,
     114,   114,   479,   479,   422,   391,   392,   393,   116,   114,
     114,   114,   430,   431,   432,   114,   499,   500,   436,   116,
      57,   504,    30,   117,   507,   114,   114,   118,   114,    27,
      28,   114,    30,    18,   114,   117,    34,    35,    36,   114,
     458,   114,   428,   117,    42,    43,    44,    45,    46,    47,
      48,   437,    50,    51,   117,    15,   117,   430,   431,   432,
      58,   117,   117,   117,    62,    57,   117,   120,    39,   114,
      15,   117,   120,    15,   460,   117,   117,    15,   114,    76,
      15,   101,   101,   101,   101,   458,    43,    24,   126,    50,
     476,   215,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,   360,   359,    19,    20,
      29,     3,    41,   499,   500,    -1,   381,    28,   504,    30,
     363,   507,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    -1,   436,    19,    20,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,   115,    19,    20,   118,    -1,    -1,
     121,    -1,   123,   124,    28,    -1,    30,    -1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,   115,    19,    20,   118,    -1,    -1,
     121,    -1,   123,   124,    28,    -1,    30,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,
      -1,    -1,   115,    -1,    -1,   118,    -1,    -1,   121,    -1,
     123,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,
      -1,   115,    -1,    -1,   118,    -1,   120,   121,    40,   123,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
      -1,   115,    -1,    -1,   118,    -1,    28,   121,    30,   123,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    28,    -1,    30,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    -1,   115,    19,    20,   118,    -1,    -1,   121,
      -1,   123,    -1,    28,    -1,    30,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    28,   115,    30,    -1,   118,    -1,    -1,   121,
      -1,   123,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   115,    -1,    -1,   118,    -1,    -1,   121,    -1,
     123,    19,    20,    -1,    22,    23,    24,    -1,    -1,    -1,
      28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     115,    -1,    -1,   118,    -1,    -1,   121,    -1,   123,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,
      -1,    -1,   118,    -1,    -1,   121,    -1,   123,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,   106,   107,
      -1,   109,   110,   111,   112,    -1,    -1,    -1,    -1,    -1,
     118,    -1,    -1,   121,    -1,   123,    -1,    -1,   126,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    -1,
      28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    61,    -1,    -1,    17,    18,    -1,    -1,
      21,    22,    23,    24,    25,    26,    -1,    -1,    -1,    -1,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    -1,    -1,
      -1,    -1,    -1,   101,    -1,   103,   104,   105,   106,   107,
      61,   109,   110,   111,   112,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   121,    -1,    -1,    -1,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    -1,    38,    -1,    -1,    -1,
     101,    -1,   103,   104,   105,   106,   107,    49,   109,   110,
     111,   112,    -1,    -1,    -1,    -1,    -1,    59,    60,    -1,
     121,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    -1,    -1,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    -1,    -1,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   158,   159,   160,     0,    31,    34,    42,    43,    44,
      45,    46,    47,    48,    62,   139,   173,   175,   177,    28,
      30,    51,    58,    62,   138,   166,   177,    45,    47,   176,
      61,    64,    65,    66,    67,    68,    69,    70,   140,   171,
      29,   183,   184,   185,    52,    53,    54,    71,   163,   113,
      61,    27,    45,    47,    50,   139,   113,   171,    30,   161,
      18,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    19,    20,   115,   118,   121,
     123,   128,   147,   148,   149,   150,   151,   166,   180,    32,
     123,   172,    33,   124,   174,   184,    72,    73,    74,    76,
      77,   138,   186,   188,   113,   113,   113,   113,   118,   164,
     161,   147,    35,    36,   157,   157,   157,   157,    18,    18,
      18,     3,   124,   151,   152,   166,   116,   125,     3,   149,
     182,     4,    15,     5,     6,     7,     8,     9,    10,    11,
      12,   134,   135,   136,    38,    49,    59,    60,    75,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   130,   131,   132,   133,   192,   193,   195,   196,
      30,    55,    56,   162,    18,    30,    30,   165,   149,   149,
     149,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,   134,   135,   137,   149,   154,   119,   119,   114,
     124,   116,    40,   152,   153,    17,    18,    21,    22,    23,
      24,    25,    26,    61,   101,   103,   109,   110,   111,   112,
     121,   130,   131,   132,   133,   155,   179,   180,   181,   181,
     181,   181,   182,    59,   140,   149,   149,   182,   149,   149,
     189,   182,   182,   182,   182,   182,   182,   149,   149,   149,
     182,   140,    99,   100,   114,   120,   114,   145,   145,   145,
      25,    26,    17,    18,   129,    18,    21,    22,    23,    24,
     118,   121,   123,   126,   155,   180,   145,   151,   151,   151,
      40,   149,   168,   169,   170,   114,   117,    63,   178,   116,
     116,   116,   116,   116,   116,   154,   156,   116,   116,   116,
     116,   114,   114,   147,   114,   142,   114,   142,   181,   118,
     114,   114,   114,   114,   114,   114,    39,   181,   181,   181,
     114,   147,   149,   182,    30,    37,    57,   143,   146,   120,
     156,   156,   124,   156,    30,   120,   122,   166,   167,   114,
     117,    40,    30,   154,   154,   154,   154,   154,   154,   114,
     122,   154,   154,   154,   154,    15,    15,   181,    10,    57,
      10,   114,   194,   181,   118,   182,   149,   182,   182,   182,
     149,   114,   114,   114,   182,   181,   181,   114,    30,    18,
     145,   120,   122,   124,    40,   168,   143,   144,   114,   194,
     114,   114,   114,   114,    39,   154,   114,   114,   114,   114,
     181,   181,   116,   181,    18,   181,   182,   190,   114,   181,
     114,   114,   114,   181,   181,   181,   116,   149,    57,   141,
      30,   117,   154,   154,   154,   154,   149,   154,   154,   154,
     154,   114,   118,   190,   191,   114,   142,   142,   114,   181,
     114,   182,   182,   182,   191,   181,    18,   114,   117,   114,
     114,   117,   117,   117,   117,   117,    15,   120,   136,   187,
     117,   182,   120,   181,   117,   154,   154,   154,   181,   179,
     120,   136,    39,   120,   117,   117,   117,   114,   179,    15,
      15,   114,   181,   181,    15,    76,   181,    15,   181
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
#line 206 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " = ";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 41:
#line 210 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string(""); 
  ;}
    break;

  case 49:
#line 217 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 56:
#line 222 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += *(yyvsp[0].String); 
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
    ;}
    break;

  case 57:
#line 227 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 58:
#line 232 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 59:
#line 233 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { *(yyvsp[-1].String) += " " + *(yyvsp[0].String); delete (yyvsp[0].String); (yyval.String) = (yyvsp[-1].String); ;}
    break;

  case 60:
#line 236 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 61:
#line 237 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, ", "); 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 62:
#line 245 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 63:
#line 251 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 65:
#line 255 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 66:
#line 256 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      (yyvsp[-1].String)->insert(0, ", ");
      if (!(yyvsp[0].String)->empty())
        *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 68:
#line 266 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 88:
#line 288 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
           (yyval.Type).newTy = (yyvsp[0].String); (yyval.Type).oldTy = OpaqueTy;
         ;}
    break;

  case 89:
#line 294 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Type UpReference
    (yyvsp[0].String)->insert(0, "\\");
    (yyval.Type).newTy = (yyvsp[0].String);
    (yyval.Type).oldTy = OpaqueTy;
  ;}
    break;

  case 90:
#line 299 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {           // Function derived type?
    *(yyvsp[-3].Type).newTy += "( " + *(yyvsp[-1].String) + " )";
    delete (yyvsp[-1].String);
    (yyval.Type).newTy = (yyvsp[-3].Type).newTy;
    (yyval.Type).oldTy = FunctionTy;
  ;}
    break;

  case 91:
#line 305 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Sized array type?
    (yyvsp[-3].String)->insert(0,"[ ");
    *(yyvsp[-3].String) += " x " + *(yyvsp[-1].Type).newTy + " ]";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].String);
    (yyval.Type).oldTy = ArrayTy;
  ;}
    break;

  case 92:
#line 312 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Packed array type?
    (yyvsp[-3].String)->insert(0,"< ");
    *(yyvsp[-3].String) += " x " + *(yyvsp[-1].Type).newTy + " >";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].String);
    (yyval.Type).oldTy = PackedTy;
  ;}
    break;

  case 93:
#line 319 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                        // Structure type?
    (yyvsp[-1].String)->insert(0, "{ ");
    *(yyvsp[-1].String) += " }";
    (yyval.Type).newTy = (yyvsp[-1].String);
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 94:
#line 325 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                  // Empty structure type?
    (yyval.Type).newTy = new std::string("{ }");
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 95:
#line 329 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                             // Pointer type?
    *(yyvsp[-1].Type).newTy += '*';
    (yyvsp[-1].Type).oldTy = PointerTy;
    (yyval.Type) = (yyvsp[-1].Type);
  ;}
    break;

  case 96:
#line 339 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].Type).newTy;
  ;}
    break;

  case 97:
#line 342 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Type).newTy;
    delete (yyvsp[0].Type).newTy;
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 99:
#line 351 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 100:
#line 356 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 101:
#line 359 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 102:
#line 369 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " [ " + *(yyvsp[-1].String) + " ]";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 103:
#line 375 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += "[ ]";
  ;}
    break;

  case 104:
#line 380 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " c" + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 105:
#line 386 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " < " + *(yyvsp[-1].String) + " >";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 106:
#line 392 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " { " + *(yyvsp[-1].String) + " }";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 107:
#line 398 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " [ ]";
  ;}
    break;

  case 108:
#line 403 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst +=  " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 109:
#line 409 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 110:
#line 415 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 111:
#line 421 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 112:
#line 427 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 113:
#line 433 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {      // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 114:
#line 439 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {            // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 115:
#line 445 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                      // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 116:
#line 451 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                     // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 117:
#line 457 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Float & Double constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 118:
#line 465 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    // We must infer the cast opcode from the types of the operands. 
    const char *opcode = getCastOpcode((yyvsp[-3].Const).type, (yyvsp[-1].Type));
    (yyval.String) = new std::string(opcode);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + " " + *(yyvsp[-2].String) + " " + *(yyvsp[-1].Type).newTy + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy();
  ;}
    break;

  case 119:
#line 472 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += "(" + *(yyvsp[-2].Const).cnst + " " + *(yyvsp[-1].String) + ")";
    (yyval.String) = (yyvsp[-4].String);
    (yyvsp[-2].Const).destroy();
    delete (yyvsp[-1].String);
  ;}
    break;

  case 120:
#line 478 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 121:
#line 483 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 122:
#line 488 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 123:
#line 493 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 124:
#line 498 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-5].String)->c_str();
    if (*(yyvsp[-5].String) == "shr")
      shiftop = ((yyvsp[-3].Const).type.isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
  ;}
    break;

  case 125:
#line 506 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 126:
#line 511 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 127:
#line 516 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 128:
#line 526 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 129:
#line 531 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Const).cnst); (yyvsp[0].Const).destroy(); ;}
    break;

  case 132:
#line 546 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
;}
    break;

  case 133:
#line 551 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 134:
#line 554 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 135:
#line 559 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "module asm " << " " << *(yyvsp[0].String) << "\n";
    (yyval.String) = 0;
  ;}
    break;

  case 136:
#line 563 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "implementation\n";
    (yyval.String) = 0;
  ;}
    break;

  case 138:
#line 570 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Type).newTy << "\n";
    // delete $2; delete $3; $4.destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 139:
#line 575 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {       // Function prototypes can be in const pool
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 140:
#line 580 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  // Asm blocks can be in the const pool
    *O << *(yyvsp[-2].String) << " " << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 141:
#line 585 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Const).cnst << " " 
       << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Const).destroy(); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 142:
#line 591 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy 
       << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 143:
#line 597 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy 
       << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 144:
#line 603 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-4].String) << " " << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy 
       << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 145:
#line 609 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 146:
#line 614 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " = " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 147:
#line 619 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = 0;
  ;}
    break;

  case 151:
#line 629 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 152:
#line 634 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    if (*(yyvsp[0].String) == "64")
      SizeOfPointer = 64;
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 153:
#line 641 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 154:
#line 646 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 155:
#line 653 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1].String)->insert(0, "[ ");
    *(yyvsp[-1].String) += " ]";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 156:
#line 660 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 158:
#line 666 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 162:
#line 675 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 163:
#line 677 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  (yyval.String) = (yyvsp[-1].Type).newTy;
  if (!(yyvsp[0].String)->empty())
    *(yyval.String) += " " + *(yyvsp[0].String);
  delete (yyvsp[0].String);
;}
    break;

  case 164:
#line 684 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 165:
#line 688 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 166:
#line 692 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 167:
#line 695 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    (yyval.String) = (yyvsp[-2].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 168:
#line 700 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 169:
#line 703 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 170:
#line 706 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 171:
#line 725 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string("begin");
  ;}
    break;

  case 172:
#line 728 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = new std::string ("{");
  ;}
    break;

  case 173:
#line 732 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if (!(yyvsp[-2].String)->empty()) {
    *O << *(yyvsp[-2].String) << " ";
  }
  *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
  delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String);
  (yyval.String) = 0;
;}
    break;

  case 174:
#line 741 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("end"); ;}
    break;

  case 175:
#line 742 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); ;}
    break;

  case 176:
#line 744 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if ((yyvsp[-1].String))
    *O << *(yyvsp[-1].String);
  *O << '\n' << *(yyvsp[0].String) << "\n";
  (yyval.String) = 0;
;}
    break;

  case 177:
#line 752 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 180:
#line 758 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[-1].String);
    *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 181:
#line 771 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 191:
#line 777 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, "<");
    *(yyvsp[-1].String) += ">";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 193:
#line 783 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty()) {
      *(yyvsp[-4].String) += " " + *(yyvsp[-3].String);
    }
    *(yyvsp[-4].String) += " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 198:
#line 801 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).type = (yyvsp[-1].Type);
    (yyval.Value).val = new std::string(*(yyvsp[-1].Type).newTy + " ");
    *(yyval.Value).val += *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 199:
#line 808 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 200:
#line 811 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Do not allow functions with 0 basic blocks   
    (yyval.String) = 0;
  ;}
    break;

  case 201:
#line 819 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 202:
#line 823 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 203:
#line 828 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 204:
#line 831 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 205:
#line 837 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {              // Return with a result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 206:
#line 842 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                       // Return with no result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Type).newTy << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 207:
#line 847 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                         // Unconditional Branch...
    *O << "    " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 208:
#line 852 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].String) << ", " 
       << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].String) << ", " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); delete (yyvsp[-6].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); 
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 209:
#line 859 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].String) << ", " << *(yyvsp[-4].Type).newTy 
       << " " << *(yyvsp[-3].String) << " [" << *(yyvsp[-1].String) << " ]\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); delete (yyvsp[-6].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 210:
#line 865 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-7].String) << " " << *(yyvsp[-6].Type).newTy << " " << *(yyvsp[-5].String) << ", " 
       << *(yyvsp[-3].Type).newTy << " " << *(yyvsp[-2].String) << "[]\n";
    delete (yyvsp[-7].String); (yyvsp[-6].Type).destroy(); delete (yyvsp[-5].String); (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String);
    (yyval.String) = 0;
  ;}
    break;

  case 211:
#line 872 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 212:
#line 884 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 213:
#line 889 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 214:
#line 895 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].String) + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 215:
#line 900 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-3].String)->insert(0, *(yyvsp[-4].Type).newTy + " " );
    *(yyvsp[-3].String) += ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 216:
#line 908 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
  ;}
    break;

  case 217:
#line 915 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {    // Used for PHI nodes
    (yyvsp[-3].String)->insert(0, *(yyvsp[-5].Type).newTy + "[");
    *(yyvsp[-3].String) += "," + *(yyvsp[-1].String) + "]";
    (yyvsp[-5].Type).destroy(); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 218:
#line 921 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6].String) += ", [" + *(yyvsp[-3].String) + "," + *(yyvsp[-1].String) + "]";
    delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 219:
#line 929 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Value).val); (yyvsp[0].Value).destroy(); ;}
    break;

  case 220:
#line 930 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 222:
#line 939 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 223:
#line 943 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 225:
#line 951 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 226:
#line 956 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 227:
#line 961 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 228:
#line 966 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 229:
#line 971 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-3].String)->c_str();
    if (*(yyvsp[-3].String) == "shr")
      shiftop = ((yyvsp[-2].Value).type.isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 230:
#line 979 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char *opcode = getCastOpcode((yyvsp[-2].Value).type, (yyvsp[0].Type));
    (yyval.String) = new std::string(opcode);
    *(yyval.String) += *(yyvsp[-2].Value).val + " " + *(yyvsp[-1].String) + " " + *(yyvsp[0].Type).newTy; 
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy();
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
  ;}
    break;

  case 231:
#line 986 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 232:
#line 991 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Type).newTy;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Type).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 233:
#line 996 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 234:
#line 1001 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 235:
#line 1006 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 236:
#line 1011 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 237:
#line 1016 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 239:
#line 1030 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[0].String)->insert(0, ", ");
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 240:
#line 1034 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  (yyval.String) = new std::string(); ;}
    break;

  case 242:
#line 1039 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 243:
#line 1042 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 244:
#line 1049 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 245:
#line 1056 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 246:
#line 1063 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 247:
#line 1070 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 248:
#line 1075 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty())
      *(yyvsp[-3].String) += " ";
    *(yyvsp[-3].String) += *(yyvsp[-2].String) + " " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 249:
#line 1082 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-5].String) += " ";
    *(yyvsp[-5].String) += *(yyvsp[-4].String) + " " + *(yyvsp[-3].Value).val + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    delete (yyvsp[-4].String); (yyvsp[-3].Value).destroy(); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 250:
#line 1089 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String) + " " + *(yyvsp[0].String);
    (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 3343 "UpgradeParser.tab.c"

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


#line 1095 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"


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

