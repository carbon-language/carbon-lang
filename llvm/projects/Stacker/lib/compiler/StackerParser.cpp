/* A Bison parser, made by GNU Bison 1.875c.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

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

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* If NAME_PREFIX is specified substitute the variables and functions
   names.  */
#define yyparse Stackerparse
#define yylex   Stackerlex
#define yyerror Stackererror
#define yylval  Stackerlval
#define yychar  Stackerchar
#define yydebug Stackerdebug
#define yynerrs Stackernerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     INTEGER = 258,
     STRING = 259,
     IDENTIFIER = 260,
     SEMI = 261,
     COLON = 262,
     FORWARD = 263,
     MAIN = 264,
     DUMP = 265,
     TRUETOK = 266,
     FALSETOK = 267,
     LESS = 268,
     MORE = 269,
     LESS_EQUAL = 270,
     MORE_EQUAL = 271,
     NOT_EQUAL = 272,
     EQUAL = 273,
     PLUS = 274,
     MINUS = 275,
     INCR = 276,
     DECR = 277,
     MULT = 278,
     DIV = 279,
     MODULUS = 280,
     NEGATE = 281,
     ABS = 282,
     MIN = 283,
     MAX = 284,
     STAR_SLASH = 285,
     AND = 286,
     OR = 287,
     XOR = 288,
     LSHIFT = 289,
     RSHIFT = 290,
     DROP = 291,
     DROP2 = 292,
     NIP = 293,
     NIP2 = 294,
     DUP = 295,
     DUP2 = 296,
     SWAP = 297,
     SWAP2 = 298,
     OVER = 299,
     OVER2 = 300,
     ROT = 301,
     ROT2 = 302,
     RROT = 303,
     RROT2 = 304,
     TUCK = 305,
     TUCK2 = 306,
     ROLL = 307,
     PICK = 308,
     SELECT = 309,
     MALLOC = 310,
     FREE = 311,
     GET = 312,
     PUT = 313,
     IF = 314,
     ELSE = 315,
     ENDIF = 316,
     WHILE = 317,
     END = 318,
     RECURSE = 319,
     RETURN = 320,
     EXIT = 321,
     TAB = 322,
     SPACE = 323,
     CR = 324,
     IN_STR = 325,
     IN_NUM = 326,
     IN_CHAR = 327,
     OUT_STR = 328,
     OUT_NUM = 329,
     OUT_CHAR = 330
   };
#endif
#define INTEGER 258
#define STRING 259
#define IDENTIFIER 260
#define SEMI 261
#define COLON 262
#define FORWARD 263
#define MAIN 264
#define DUMP 265
#define TRUETOK 266
#define FALSETOK 267
#define LESS 268
#define MORE 269
#define LESS_EQUAL 270
#define MORE_EQUAL 271
#define NOT_EQUAL 272
#define EQUAL 273
#define PLUS 274
#define MINUS 275
#define INCR 276
#define DECR 277
#define MULT 278
#define DIV 279
#define MODULUS 280
#define NEGATE 281
#define ABS 282
#define MIN 283
#define MAX 284
#define STAR_SLASH 285
#define AND 286
#define OR 287
#define XOR 288
#define LSHIFT 289
#define RSHIFT 290
#define DROP 291
#define DROP2 292
#define NIP 293
#define NIP2 294
#define DUP 295
#define DUP2 296
#define SWAP 297
#define SWAP2 298
#define OVER 299
#define OVER2 300
#define ROT 301
#define ROT2 302
#define RROT 303
#define RROT2 304
#define TUCK 305
#define TUCK2 306
#define ROLL 307
#define PICK 308
#define SELECT 309
#define MALLOC 310
#define FREE 311
#define GET 312
#define PUT 313
#define IF 314
#define ELSE 315
#define ENDIF 316
#define WHILE 317
#define END 318
#define RECURSE 319
#define RETURN 320
#define EXIT 321
#define TAB 322
#define SPACE 323
#define CR 324
#define IN_STR 325
#define IN_NUM 326
#define IN_CHAR 327
#define OUT_STR 328
#define OUT_NUM 329
#define OUT_CHAR 330




/* Copy the first part of user declarations.  */
#line 14 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"

#include "StackerCompiler.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include <list>
#include <utility>
#include <algorithm>

#define YYERROR_VERBOSE 1
#define SCI StackerCompiler::TheInstance

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit
int yylex();                       // declaration" of xxx warnings.
int yyparse();



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

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 35 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
typedef union YYSTYPE {
  llvm::Module*		ModuleVal;
  llvm::Function* 	FunctionVal;
  llvm::BasicBlock*	BasicBlockVal;
  int64_t               IntegerVal;
  char*                 StringVal;
} YYSTYPE;
/* Line 191 of yacc.c.  */
#line 263 "StackerParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 275 "StackerParser.tab.c"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

# ifndef YYFREE
#  define YYFREE free
# endif
# ifndef YYMALLOC
#  define YYMALLOC malloc
# endif

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   define YYSTACK_ALLOC alloca
#  endif
# else
#  if defined (alloca) || defined (_ALLOCA_H)
#   define YYSTACK_ALLOC alloca
#  else
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (defined (YYSTYPE_IS_TRIVIAL) && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short) + sizeof (YYSTYPE))				\
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
	  register YYSIZE_T yyi;		\
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
   typedef short yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   150

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  10
/* YYNRULES -- Number of rules. */
#define YYNRULES  80
/* YYNRULES -- Number of states. */
#define YYNSTATES  93

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   330

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
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
      75
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned char yyprhs[] =
{
       0,     0,     3,     4,     7,    10,    11,    13,    15,    17,
      21,    26,    31,    34,    35,    41,    45,    49,    51,    53,
      55,    57,    59,    61,    63,    65,    67,    69,    71,    73,
      75,    77,    79,    81,    83,    85,    87,    89,    91,    93,
      95,    97,    99,   101,   103,   105,   107,   109,   111,   113,
     115,   117,   119,   121,   123,   125,   127,   129,   131,   133,
     135,   137,   139,   141,   143,   145,   147,   149,   151,   153,
     155,   157,   159,   161,   163,   165,   167,   169,   171,   173,
     175
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const yysigned_char yyrhs[] =
{
      77,     0,    -1,    -1,    78,    79,    -1,    79,    80,    -1,
      -1,    81,    -1,    83,    -1,    82,    -1,     8,     5,     6,
      -1,     7,     9,    84,     6,    -1,     7,     5,    84,     6,
      -1,    84,    85,    -1,    -1,    59,     5,    60,     5,    61,
      -1,    59,     5,    61,    -1,    62,     5,    63,    -1,     5,
      -1,     4,    -1,     3,    -1,    11,    -1,    12,    -1,    13,
      -1,    14,    -1,    15,    -1,    16,    -1,    17,    -1,    18,
      -1,    19,    -1,    20,    -1,    21,    -1,    22,    -1,    23,
      -1,    24,    -1,    25,    -1,    26,    -1,    27,    -1,    28,
      -1,    29,    -1,    30,    -1,    31,    -1,    32,    -1,    33,
      -1,    34,    -1,    35,    -1,    36,    -1,    37,    -1,    38,
      -1,    39,    -1,    40,    -1,    41,    -1,    42,    -1,    43,
      -1,    44,    -1,    45,    -1,    46,    -1,    47,    -1,    48,
      -1,    49,    -1,    50,    -1,    51,    -1,    52,    -1,    53,
      -1,    54,    -1,    55,    -1,    56,    -1,    57,    -1,    58,
      -1,    64,    -1,    65,    -1,    66,    -1,    67,    -1,    68,
      -1,    69,    -1,    70,    -1,    71,    -1,    72,    -1,    73,
      -1,    74,    -1,    75,    -1,    10,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned char yyrline[] =
{
       0,    70,    70,    70,    74,    75,    78,    79,    80,    83,
      86,    89,    92,    93,    99,   100,   101,   104,   105,   106,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "INTEGER", "STRING", "IDENTIFIER",
  "SEMI", "COLON", "FORWARD", "MAIN", "DUMP", "TRUETOK", "FALSETOK",
  "LESS", "MORE", "LESS_EQUAL", "MORE_EQUAL", "NOT_EQUAL", "EQUAL", "PLUS",
  "MINUS", "INCR", "DECR", "MULT", "DIV", "MODULUS", "NEGATE", "ABS",
  "MIN", "MAX", "STAR_SLASH", "AND", "OR", "XOR", "LSHIFT", "RSHIFT",
  "DROP", "DROP2", "NIP", "NIP2", "DUP", "DUP2", "SWAP", "SWAP2", "OVER",
  "OVER2", "ROT", "ROT2", "RROT", "RROT2", "TUCK", "TUCK2", "ROLL", "PICK",
  "SELECT", "MALLOC", "FREE", "GET", "PUT", "IF", "ELSE", "ENDIF", "WHILE",
  "END", "RECURSE", "RETURN", "EXIT", "TAB", "SPACE", "CR", "IN_STR",
  "IN_NUM", "IN_CHAR", "OUT_STR", "OUT_NUM", "OUT_CHAR", "$accept",
  "Module", "@1", "DefinitionList", "Definition", "ForwardDef", "MainDef",
  "ColonDef", "WordList", "Word", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    76,    78,    77,    79,    79,    80,    80,    80,    81,
      82,    83,    84,    84,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    85,
      85
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     0,     2,     2,     0,     1,     1,     1,     3,
       4,     4,     2,     0,     5,     3,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       2,     0,     5,     1,     3,     0,     0,     4,     6,     8,
       7,    13,    13,     0,     0,     0,     9,    19,    18,    17,
      11,    80,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
       0,     0,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    12,    10,     0,     0,     0,    15,
      16,     0,    14
};

/* YYDEFGOTO[NTERM-NUM]. */
static const yysigned_char yydefgoto[] =
{
      -1,     1,     2,     4,     7,     8,     9,    10,    14,    84
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -4
static const short yypact[] =
{
      -4,     4,    -4,    -4,    -2,   141,    52,    -4,    -4,    -4,
      -4,    -4,    -4,    54,    -3,    70,    -4,    -4,    -4,    -4,
      -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,
      -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,
      -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,
      -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,
      -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,
      53,    74,    -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,
      -4,    -4,    -4,    -4,    -4,    -4,    17,    67,   126,    -4,
      -4,    72,    -4
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
      -4,    -4,    -4,    -4,    -4,    -4,    -4,    -4,   135,    -4
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const unsigned char yytable[] =
{
      17,    18,    19,    20,     3,     5,     6,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    13,    86,    71,
      16,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    17,    18,    19,    85,    88,    89,    87,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      90,    91,    71,    92,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    11,    15,     0,     0,
      12
};

static const yysigned_char yycheck[] =
{
       3,     4,     5,     6,     0,     7,     8,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,     5,     5,    62,
       6,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,     3,     4,     5,     6,    60,    61,     5,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      63,     5,    62,    61,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,     5,    12,    -1,    -1,
       9
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,    77,    78,     0,    79,     7,     8,    80,    81,    82,
      83,     5,     9,     5,    84,    84,     6,     3,     4,     5,
       6,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    62,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    85,     6,     5,     5,    60,    61,
      63,     5,    61
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

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
    { 								\
      yyerror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)		\
   ((Current).first_line   = (Rhs)[1].first_line,	\
    (Current).first_column = (Rhs)[1].first_column,	\
    (Current).last_line    = (Rhs)[N].last_line,	\
    (Current).last_column  = (Rhs)[N].last_column)
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

# define YYDSYMPRINT(Args)			\
do {						\
  if (yydebug)					\
    yysymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short *bottom, short *top)
#else
static void
yy_stack_print (bottom, top)
    short *bottom;
    short *top;
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
  unsigned int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname [yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname [yyr1[yyrule]]);
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
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
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
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if defined (YYMAXDEPTH) && YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

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
  register const char *yys = yystr;

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
  register char *yyd = yydest;
  register const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



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
    {
      YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
# ifdef YYPRINT
      YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
    }
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

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
yydestruct (int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yytype, yyvaluep)
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

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



/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
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
  
  register int yystate;
  register int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short	yyssa[YYINITDEPTH];
  short *yyss = yyssa;
  register short *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  register YYSTYPE *yyvsp;



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
	short *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow ("parser stack overflow",
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyoverflowlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyoverflowlab;
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
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
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
      YYDSYMPRINTF ("Next token is", yytoken, &yylval, &yylloc);
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

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", yytname[yytoken]));

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
        case 2:
#line 70 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { SCI->handle_module_start( ); ;}
    break;

  case 3:
#line 71 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.ModuleVal = SCI->handle_module_end( yyvsp[0].ModuleVal ); ;}
    break;

  case 4:
#line 74 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.ModuleVal = SCI->handle_definition_list_end( yyvsp[-1].ModuleVal, yyvsp[0].FunctionVal ); ;}
    break;

  case 5:
#line 75 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.ModuleVal = SCI->handle_definition_list_start(); ;}
    break;

  case 6:
#line 78 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = yyvsp[0].FunctionVal; ;}
    break;

  case 7:
#line 79 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = yyvsp[0].FunctionVal; ;}
    break;

  case 8:
#line 80 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = yyvsp[0].FunctionVal; ;}
    break;

  case 9:
#line 83 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = SCI->handle_forward( yyvsp[-1].StringVal ); ;}
    break;

  case 10:
#line 86 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = SCI->handle_main_definition(yyvsp[-1].FunctionVal); ;}
    break;

  case 11:
#line 89 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = SCI->handle_definition( yyvsp[-2].StringVal, yyvsp[-1].FunctionVal ); ;}
    break;

  case 12:
#line 92 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = SCI->handle_word_list_end( yyvsp[-1].FunctionVal, yyvsp[0].BasicBlockVal ); ;}
    break;

  case 13:
#line 93 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.FunctionVal = SCI->handle_word_list_start(); ;}
    break;

  case 14:
#line 99 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_if( yyvsp[-3].StringVal, yyvsp[-1].StringVal ); ;}
    break;

  case 15:
#line 100 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_if( yyvsp[-1].StringVal ); ;}
    break;

  case 16:
#line 101 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_while( yyvsp[-1].StringVal ); ;}
    break;

  case 17:
#line 104 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_identifier( yyvsp[0].StringVal ); ;}
    break;

  case 18:
#line 105 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_string( yyvsp[0].StringVal ); ;}
    break;

  case 19:
#line 106 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_integer( yyvsp[0].IntegerVal ); ;}
    break;

  case 20:
#line 109 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( TRUETOK ); ;}
    break;

  case 21:
#line 110 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( FALSETOK ); ;}
    break;

  case 22:
#line 111 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( LESS ); ;}
    break;

  case 23:
#line 112 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MORE ); ;}
    break;

  case 24:
#line 113 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( LESS_EQUAL ); ;}
    break;

  case 25:
#line 114 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MORE_EQUAL ); ;}
    break;

  case 26:
#line 115 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( NOT_EQUAL ); ;}
    break;

  case 27:
#line 116 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( EQUAL ); ;}
    break;

  case 28:
#line 117 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( PLUS ); ;}
    break;

  case 29:
#line 118 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MINUS ); ;}
    break;

  case 30:
#line 119 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( INCR ); ;}
    break;

  case 31:
#line 120 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DECR ); ;}
    break;

  case 32:
#line 121 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MULT ); ;}
    break;

  case 33:
#line 122 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DIV ); ;}
    break;

  case 34:
#line 123 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MODULUS ); ;}
    break;

  case 35:
#line 124 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( NEGATE ); ;}
    break;

  case 36:
#line 125 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( ABS ); ;}
    break;

  case 37:
#line 126 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MIN ); ;}
    break;

  case 38:
#line 127 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MAX ); ;}
    break;

  case 39:
#line 128 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( STAR_SLASH ); ;}
    break;

  case 40:
#line 129 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( AND ); ;}
    break;

  case 41:
#line 130 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( OR ); ;}
    break;

  case 42:
#line 131 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( XOR ); ;}
    break;

  case 43:
#line 132 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( LSHIFT ); ;}
    break;

  case 44:
#line 133 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( RSHIFT ); ;}
    break;

  case 45:
#line 134 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DROP ); ;}
    break;

  case 46:
#line 135 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DROP2 ); ;}
    break;

  case 47:
#line 136 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( NIP ); ;}
    break;

  case 48:
#line 137 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( NIP2 ); ;}
    break;

  case 49:
#line 138 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DUP ); ;}
    break;

  case 50:
#line 139 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DUP2 ); ;}
    break;

  case 51:
#line 140 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( SWAP ); ;}
    break;

  case 52:
#line 141 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( SWAP2 ); ;}
    break;

  case 53:
#line 142 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( OVER ); ;}
    break;

  case 54:
#line 143 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( OVER2 ); ;}
    break;

  case 55:
#line 144 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( ROT ); ;}
    break;

  case 56:
#line 145 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( ROT2 ); ;}
    break;

  case 57:
#line 146 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( RROT ); ;}
    break;

  case 58:
#line 147 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( RROT2 ); ;}
    break;

  case 59:
#line 148 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( TUCK ); ;}
    break;

  case 60:
#line 149 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( TUCK2 ); ;}
    break;

  case 61:
#line 150 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( ROLL ); ;}
    break;

  case 62:
#line 151 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( PICK ); ;}
    break;

  case 63:
#line 152 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( SELECT ); ;}
    break;

  case 64:
#line 153 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( MALLOC ); ;}
    break;

  case 65:
#line 154 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( FREE ); ;}
    break;

  case 66:
#line 155 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( GET ); ;}
    break;

  case 67:
#line 156 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( PUT ); ;}
    break;

  case 68:
#line 157 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( RECURSE ); ;}
    break;

  case 69:
#line 158 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( RETURN ); ;}
    break;

  case 70:
#line 159 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( EXIT ); ;}
    break;

  case 71:
#line 160 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( TAB ); ;}
    break;

  case 72:
#line 161 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( SPACE ); ;}
    break;

  case 73:
#line 162 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( CR ); ;}
    break;

  case 74:
#line 163 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( IN_STR ); ;}
    break;

  case 75:
#line 164 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( IN_NUM ); ;}
    break;

  case 76:
#line 165 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( IN_CHAR ); ;}
    break;

  case 77:
#line 166 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( OUT_STR ); ;}
    break;

  case 78:
#line 167 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( OUT_NUM ); ;}
    break;

  case 79:
#line 168 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( OUT_CHAR ); ;}
    break;

  case 80:
#line 169 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"
    { yyval.BasicBlockVal = SCI->handle_word( DUMP ); ;}
    break;


    }

/* Line 1000 of yacc.c.  */
#line 1669 "StackerParser.tab.c"

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
	  YYSIZE_T yysize = 0;
	  int yytype = YYTRANSLATE (yychar);
	  const char* yyprefix;
	  char *yymsg;
	  int yyx;

	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  int yyxbegin = yyn < 0 ? -yyn : 0;

	  /* Stay within bounds of both yycheck and yytname.  */
	  int yychecklim = YYLAST - yyn;
	  int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
	  int yycount = 0;

	  yyprefix = ", expecting ";
	  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      {
		yysize += yystrlen (yyprefix) + yystrlen (yytname [yyx]);
		yycount += 1;
		if (yycount == 5)
		  {
		    yysize = 0;
		    break;
		  }
	      }
	  yysize += (sizeof ("syntax error, unexpected ")
		     + yystrlen (yytname[yytype]));
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yyprefix = ", expecting ";
		  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			yyp = yystpcpy (yyp, yyprefix);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yyprefix = " or ";
		      }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    yyerror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror ("syntax error");
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* If at end of input, pop the error token,
	     then the rest of the stack, then return failure.  */
	  if (yychar == YYEOF)
	     for (;;)
	       {
		 YYPOPSTACK;
		 if (yyssp == yyss)
		   YYABORT;
		 YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
		 yydestruct (yystos[*yyssp], yyvsp);
	       }
        }
      else
	{
	  YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
	  yydestruct (yytoken, &yylval);
	  yychar = YYEMPTY;

	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

#ifdef __GNUC__
  /* Pacify GCC when the user code never invokes YYERROR and the label
     yyerrorlab therefore never appears in user code.  */
  if (0)
     goto yyerrorlab;
#endif

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

      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
      yydestruct (yystos[yystate], yyvsp);
      YYPOPSTACK;
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++yyvsp = yylval;


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
/*----------------------------------------------.
| yyoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
yyoverflowlab:
  yyerror ("parser stack overflow");
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 171 "/proj/llvm/build/projects/Stacker/../../../llvm/projects/Stacker/lib/compiler/StackerParser.y"


/* Handle messages a little more nicely than the default yyerror */
int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((SCI->filename() == "-") ? std::string("<stdin>") : SCI->filename())
                  + ":" + utostr((unsigned) Stackerlineno ) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Stackertext, Stackerleng) + "'";
  StackerCompiler::ThrowException(errMsg);
  return 0;
}

