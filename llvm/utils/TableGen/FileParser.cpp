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
#define yyparse Fileparse
#define yylex   Filelex
#define yyerror Fileerror
#define yylval  Filelval
#define yychar  Filechar
#define yydebug Filedebug
#define yynerrs Filenerrs


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




/* Copy the first part of user declarations.  */
#line 14 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"

#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cstdio>
#define YYERROR_VERBOSE 1

int yyerror(const char *ErrorMsg);
int yylex();

namespace llvm {

extern int Filelineno;
static Record *CurRec = 0;
static bool ParsingTemplateArgs = false;

typedef std::pair<Record*, std::vector<Init*>*> SubClassRefTy;

struct LetRecord {
  std::string Name;
  std::vector<unsigned> Bits;
  Init *Value;
  bool HasBits;
  LetRecord(const std::string &N, std::vector<unsigned> *B, Init *V)
    : Name(N), Value(V), HasBits(B != 0) {
    if (HasBits) Bits = *B;
  }
};

static std::vector<std::vector<LetRecord> > LetStack;


extern std::ostream &err();

static void addValue(const RecordVal &RV) {
  if (RecordVal *ERV = CurRec->getValue(RV.getName())) {
    // The value already exists in the class, treat this as a set...
    if (ERV->setValue(RV.getValue())) {
      err() << "New definition of '" << RV.getName() << "' of type '"
            << *RV.getType() << "' is incompatible with previous "
            << "definition of type '" << *ERV->getType() << "'!\n";
      exit(1);
    }
  } else {
    CurRec->addValue(RV);
  }
}

static void addSuperClass(Record *SC) {
  if (CurRec->isSubClassOf(SC)) {
    err() << "Already subclass of '" << SC->getName() << "'!\n";
    exit(1);
  }
  CurRec->addSuperClass(SC);
}

static void setValue(const std::string &ValName, 
		     std::vector<unsigned> *BitList, Init *V) {
  if (!V) return;

  RecordVal *RV = CurRec->getValue(ValName);
  if (RV == 0) {
    err() << "Value '" << ValName << "' unknown!\n";
    exit(1);
  }

  // Do not allow assignments like 'X = X'.  This will just cause infinite loops
  // in the resolution machinery.
  if (!BitList)
    if (VarInit *VI = dynamic_cast<VarInit*>(V))
      if (VI->getName() == ValName)
        return;
  
  // If we are assigning to a subset of the bits in the value... then we must be
  // assigning to a field of BitsRecTy, which must have a BitsInit
  // initializer...
  //
  if (BitList) {
    BitsInit *CurVal = dynamic_cast<BitsInit*>(RV->getValue());
    if (CurVal == 0) {
      err() << "Value '" << ValName << "' is not a bits type!\n";
      exit(1);
    }

    // Convert the incoming value to a bits type of the appropriate size...
    Init *BI = V->convertInitializerTo(new BitsRecTy(BitList->size()));
    if (BI == 0) {
      V->convertInitializerTo(new BitsRecTy(BitList->size()));
      err() << "Initializer '" << *V << "' not compatible with bit range!\n";
      exit(1);
    }

    // We should have a BitsInit type now...
    assert(dynamic_cast<BitsInit*>(BI) != 0 || &(std::cerr << *BI) == 0);
    BitsInit *BInit = (BitsInit*)BI;

    BitsInit *NewVal = new BitsInit(CurVal->getNumBits());

    // Loop over bits, assigning values as appropriate...
    for (unsigned i = 0, e = BitList->size(); i != e; ++i) {
      unsigned Bit = (*BitList)[i];
      if (NewVal->getBit(Bit)) {
        err() << "Cannot set bit #" << Bit << " of value '" << ValName
              << "' more than once!\n";
        exit(1);
      }
      NewVal->setBit(Bit, BInit->getBit(i));
    }

    for (unsigned i = 0, e = CurVal->getNumBits(); i != e; ++i)
      if (NewVal->getBit(i) == 0)
        NewVal->setBit(i, CurVal->getBit(i));

    V = NewVal;
  }

  if (RV->setValue(V)) {
    err() << "Value '" << ValName << "' of type '" << *RV->getType()
	  << "' is incompatible with initializer '" << *V << "'!\n";
    exit(1);
  }
}

// addSubClass - Add SC as a subclass to CurRec, resolving TemplateArgs as SC's
// template arguments.
static void addSubClass(Record *SC, const std::vector<Init*> &TemplateArgs) {
  // Add all of the values in the subclass into the current class...
  const std::vector<RecordVal> &Vals = SC->getValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    addValue(Vals[i]);

  const std::vector<std::string> &TArgs = SC->getTemplateArgs();

  // Ensure that an appropriate number of template arguments are specified...
  if (TArgs.size() < TemplateArgs.size()) {
    err() << "ERROR: More template args specified than expected!\n";
    exit(1);
  } else {    // This class expects template arguments...
    // Loop over all of the template arguments, setting them to the specified
    // value or leaving them as the default if necessary.
    for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
      if (i < TemplateArgs.size()) {  // A value is specified for this temp-arg?
        // Set it now.
        setValue(TArgs[i], 0, TemplateArgs[i]);

        // Resolve it next.
        CurRec->resolveReferencesTo(CurRec->getValue(TArgs[i]));
                                    
        
        // Now remove it.
        CurRec->removeValue(TArgs[i]);

      } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
        err() << "ERROR: Value not specified for template argument #"
              << i << " (" << TArgs[i] << ") of subclass '" << SC->getName()
              << "'!\n";
        exit(1);
      }
    }
  }

  // Since everything went well, we can now set the "superclass" list for the
  // current record.
  const std::vector<Record*> &SCs  = SC->getSuperClasses();
  for (unsigned i = 0, e = SCs.size(); i != e; ++i)
    addSuperClass(SCs[i]);
  addSuperClass(SC);
}

} // End llvm namespace

using namespace llvm;



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
#line 189 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
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
/* Line 191 of yacc.c.  */
#line 316 "FileParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 328 "FileParser.tab.c"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# if YYSTACK_USE_ALLOCA
#  define YYSTACK_ALLOC alloca
# else
#  ifndef YYSTACK_USE_ALLOCA
#   if defined (alloca) || defined (_ALLOCA_H)
#    define YYSTACK_ALLOC alloca
#   else
#    ifdef __GNUC__
#     define YYSTACK_ALLOC __builtin_alloca
#    endif
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
#  define YYSTACK_ALLOC malloc
#  define YYSTACK_FREE free
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (YYSTYPE_IS_TRIVIAL)))

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
#  if 1 < __GNUC__
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
#define YYFINAL  18
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   153

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  38
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  37
/* YYNRULES -- Number of rules. */
#define YYNRULES  85
/* YYNRULES -- Number of states. */
#define YYNSTATES  155

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   277

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      32,    33,     2,     2,    34,    36,    31,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    35,    37,
      23,    25,    24,    26,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    29,     2,    30,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    27,     2,    28,     2,     2,     2,     2,
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
      15,    16,    17,    18,    19,    20,    21,    22
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    14,    16,    21,    23,
      25,    27,    28,    30,    31,    34,    36,    38,    40,    42,
      46,    51,    53,    58,    62,    66,    71,    76,    83,    90,
      97,    98,   101,   104,   109,   110,   112,   114,   118,   121,
     125,   131,   136,   138,   139,   143,   144,   146,   148,   152,
     157,   160,   167,   168,   171,   173,   177,   179,   184,   186,
     190,   191,   194,   196,   200,   204,   205,   207,   209,   210,
     211,   212,   219,   222,   225,   227,   229,   234,   236,   240,
     241,   246,   251,   254,   256,   259
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const yysigned_char yyrhs[] =
{
      74,     0,    -1,    19,    -1,     5,    -1,     4,    -1,     6,
      23,    18,    24,    -1,     3,    -1,     7,    23,    40,    24,
      -1,     8,    -1,     9,    -1,    39,    -1,    -1,    12,    -1,
      -1,    25,    43,    -1,    18,    -1,    21,    -1,    22,    -1,
      26,    -1,    27,    50,    28,    -1,    19,    23,    51,    24,
      -1,    19,    -1,    43,    27,    48,    28,    -1,    29,    50,
      30,    -1,    43,    31,    19,    -1,    32,    19,    46,    33,
      -1,    43,    29,    48,    30,    -1,    15,    32,    43,    34,
      43,    33,    -1,    16,    32,    43,    34,    43,    33,    -1,
      17,    32,    43,    34,    43,    33,    -1,    -1,    35,    20,
      -1,    43,    44,    -1,    45,    34,    43,    44,    -1,    -1,
      45,    -1,    18,    -1,    18,    36,    18,    -1,    18,    18,
      -1,    47,    34,    18,    -1,    47,    34,    18,    36,    18,
      -1,    47,    34,    18,    18,    -1,    47,    -1,    -1,    27,
      48,    28,    -1,    -1,    51,    -1,    43,    -1,    51,    34,
      43,    -1,    41,    40,    19,    42,    -1,    52,    37,    -1,
      13,    19,    49,    25,    43,    37,    -1,    -1,    54,    53,
      -1,    37,    -1,    27,    54,    28,    -1,    39,    -1,    39,
      23,    51,    24,    -1,    56,    -1,    57,    34,    56,    -1,
      -1,    35,    57,    -1,    52,    -1,    59,    34,    52,    -1,
      23,    59,    24,    -1,    -1,    60,    -1,    19,    -1,    -1,
      -1,    -1,    62,    64,    61,    58,    65,    55,    -1,    10,
      63,    -1,    11,    63,    -1,    66,    -1,    67,    -1,    19,
      49,    25,    43,    -1,    69,    -1,    70,    34,    69,    -1,
      -1,    13,    72,    70,    14,    -1,    71,    27,    73,    28,
      -1,    71,    68,    -1,    68,    -1,    73,    68,    -1,    73,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   223,   223,   234,   236,   238,   240,   242,   244,   246,
     248,   252,   252,   254,   254,   256,   258,   261,   264,   266,
     279,   307,   322,   329,   332,   339,   347,   355,   361,   367,
     375,   378,   382,   387,   393,   396,   399,   402,   415,   429,
     431,   444,   460,   462,   462,   466,   468,   472,   475,   479,
     489,   491,   497,   497,   498,   498,   500,   502,   506,   511,
     516,   519,   523,   526,   531,   532,   532,   534,   534,   536,
     543,   536,   563,   571,   589,   589,   591,   596,   596,   599,
     599,   602,   605,   609,   609,   611
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "INT", "BIT", "STRING", "BITS", "LIST", 
  "CODE", "DAG", "CLASS", "DEF", "FIELD", "LET", "IN", "SHLTOK", "SRATOK", 
  "SRLTOK", "INTVAL", "ID", "VARNAME", "STRVAL", "CODEFRAGMENT", "'<'", 
  "'>'", "'='", "'?'", "'{'", "'}'", "'['", "']'", "'.'", "'('", "')'", 
  "','", "':'", "'-'", "';'", "$accept", "ClassID", "Type", "OptPrefix", 
  "OptValue", "Value", "OptVarName", "DagArgListNE", "DagArgList", 
  "RBitList", "BitList", "OptBitList", "ValueList", "ValueListNE", 
  "Declaration", "BodyItem", "BodyList", "Body", "SubClassRef", 
  "ClassListNE", "ClassList", "DeclListNE", "TemplateArgList", 
  "OptTemplateArgList", "OptID", "ObjectBody", "@1", "@2", "ClassInst", 
  "DefInst", "Object", "LETItem", "LETList", "LETCommand", "@3", 
  "ObjectList", "File", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,    60,    62,    61,    63,   123,   125,    91,
      93,    46,    40,    41,    44,    58,    45,    59
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    38,    39,    40,    40,    40,    40,    40,    40,    40,
      40,    41,    41,    42,    42,    43,    43,    43,    43,    43,
      43,    43,    43,    43,    43,    43,    43,    43,    43,    43,
      44,    44,    45,    45,    46,    46,    47,    47,    47,    47,
      47,    47,    48,    49,    49,    50,    50,    51,    51,    52,
      53,    53,    54,    54,    55,    55,    56,    56,    57,    57,
      58,    58,    59,    59,    60,    61,    61,    62,    62,    64,
      65,    63,    66,    67,    68,    68,    69,    70,    70,    72,
      71,    68,    68,    73,    73,    74
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     4,     1,     4,     1,     1,
       1,     0,     1,     0,     2,     1,     1,     1,     1,     3,
       4,     1,     4,     3,     3,     4,     4,     6,     6,     6,
       0,     2,     2,     4,     0,     1,     1,     3,     2,     3,
       5,     4,     1,     0,     3,     0,     1,     1,     3,     4,
       2,     6,     0,     2,     1,     3,     1,     4,     1,     3,
       0,     2,     1,     3,     3,     0,     1,     1,     0,     0,
       0,     6,     2,     2,     1,     1,     4,     1,     3,     0,
       4,     4,     2,     1,     2,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       0,    68,    68,    79,    74,    75,    83,     0,    85,     0,
      67,    69,    72,    73,     0,     0,    82,    84,     1,    65,
      43,    77,     0,     0,    11,    66,    60,     0,     0,    80,
       0,    81,    12,     0,    62,     0,     0,    70,    36,    42,
       0,     0,    78,     6,     4,     3,     0,     0,     8,     9,
       2,    10,     0,    64,    11,    56,    58,    61,     0,    38,
       0,     0,    44,     0,     0,     0,    15,    21,    16,    17,
      18,    45,    45,     0,    76,     0,     0,    13,    63,     0,
       0,    52,    54,    71,    37,    39,     0,     0,     0,     0,
      47,     0,    46,     0,    34,     0,     0,     0,     0,     0,
       0,    49,     0,    59,    11,    41,     0,     0,     0,     0,
       0,    19,     0,    23,    30,    35,     0,     0,     0,    24,
       5,     7,    14,    57,     0,    55,     0,    53,    40,     0,
       0,     0,    20,    48,     0,    32,     0,    25,    22,    26,
      43,    50,     0,     0,     0,    31,    30,     0,    27,    28,
      29,    33,     0,     0,    51
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,    51,    52,    33,   101,    90,   135,   115,   116,    39,
      40,    28,    91,    92,    34,   127,   104,    83,    56,    57,
      37,    35,    25,    26,    11,    12,    19,    58,     4,     5,
       6,    21,    22,     7,    14,     8,     9
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -61
static const yysigned_char yypact[] =
{
      45,    -7,    -7,   -61,   -61,   -61,   -61,     4,    45,    24,
     -61,   -61,   -61,   -61,    31,    45,   -61,   -61,   -61,    29,
      46,   -61,   -12,    -5,    62,   -61,    40,    63,    68,   -61,
      31,   -61,   -61,    57,   -61,   -14,    78,   -61,   -15,    67,
      75,    11,   -61,   -61,   -61,   -61,    84,    96,   -61,   -61,
     -61,   -61,    90,   -61,    62,    97,   -61,    87,    12,   -61,
     104,   105,   -61,    92,    93,    94,   -61,   106,   -61,   -61,
     -61,    11,    11,   108,    86,   110,    57,   107,   -61,    11,
      78,   -61,   -61,   -61,   -61,   -11,    11,    11,    11,    11,
      86,   102,    99,   101,    11,    63,    63,   115,   111,   112,
      11,   -61,    20,   -61,     6,   -61,   119,   -18,    65,    71,
      43,   -61,    11,   -61,    56,   109,   113,   114,   117,   -61,
     -61,   -61,    86,   -61,   120,   -61,   103,   -61,   -61,    11,
      11,    11,   -61,    86,   118,   -61,    11,   -61,   -61,   -61,
      46,   -61,    53,    77,    85,   -61,    56,   116,   -61,   -61,
     -61,   -61,    11,    41,   -61
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
     -61,   -32,    69,   -61,   -61,   -41,    -2,   -61,   -61,   -61,
     -60,     8,    79,   -10,   -53,   -61,   -61,   -61,    70,   -61,
     -61,   -61,   -61,   -61,   -61,   147,   -61,   -61,   -61,   -61,
      34,   122,   -61,   -61,   -61,   138,   -61
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const unsigned char yytable[] =
{
      74,    78,    29,    59,    55,     1,     2,   105,     3,    95,
      53,    96,    10,    97,     1,     2,   129,     3,    32,   124,
      54,    60,    30,    31,    18,   106,    63,    64,    65,    66,
      67,    15,    68,    69,   125,   117,   118,    70,    71,    81,
      72,    16,    17,    73,   123,   107,   108,   109,    55,    82,
      20,   126,    24,   114,   112,     1,     2,    17,     3,   122,
      43,    44,    45,    46,    47,    48,    49,   132,    95,   102,
      96,   133,    97,    27,    32,    36,    50,   112,   154,   110,
      95,    38,    96,    95,    97,    96,   148,    97,   142,   143,
     144,   134,    95,    41,    96,   146,    97,    50,    95,   130,
      96,    61,    97,    62,    95,   131,    96,    75,    97,    77,
     149,   153,    95,    95,    96,    96,    97,    97,   150,    76,
      79,    80,    84,    85,    86,    87,    88,    94,    98,    89,
     111,   113,   100,   112,   119,   120,   121,   128,   145,   140,
     141,   152,   138,   136,   151,    99,   137,   139,   147,    13,
     103,    93,    42,    23
};

static const unsigned char yycheck[] =
{
      41,    54,    14,    18,    36,    10,    11,    18,    13,    27,
      24,    29,    19,    31,    10,    11,    34,    13,    12,    13,
      34,    36,    34,    28,     0,    36,    15,    16,    17,    18,
      19,    27,    21,    22,    28,    95,    96,    26,    27,    27,
      29,     7,     8,    32,    24,    86,    87,    88,    80,    37,
      19,   104,    23,    94,    34,    10,    11,    23,    13,   100,
       3,     4,     5,     6,     7,     8,     9,    24,    27,    79,
      29,   112,    31,    27,    12,    35,    19,    34,    37,    89,
      27,    18,    29,    27,    31,    29,    33,    31,   129,   130,
     131,    35,    27,    25,    29,   136,    31,    19,    27,    34,
      29,    34,    31,    28,    27,    34,    29,    23,    31,    19,
      33,   152,    27,    27,    29,    29,    31,    31,    33,    23,
      23,    34,    18,    18,    32,    32,    32,    19,    18,    23,
      28,    30,    25,    34,    19,    24,    24,    18,    20,    19,
      37,    25,    28,    34,   146,    76,    33,    30,   140,     2,
      80,    72,    30,    15
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,    10,    11,    13,    66,    67,    68,    71,    73,    74,
      19,    62,    63,    63,    72,    27,    68,    68,     0,    64,
      19,    69,    70,    73,    23,    60,    61,    27,    49,    14,
      34,    28,    12,    41,    52,    59,    35,    58,    18,    47,
      48,    25,    69,     3,     4,     5,     6,     7,     8,     9,
      19,    39,    40,    24,    34,    39,    56,    57,    65,    18,
      36,    34,    28,    15,    16,    17,    18,    19,    21,    22,
      26,    27,    29,    32,    43,    23,    23,    19,    52,    23,
      34,    27,    37,    55,    18,    18,    32,    32,    32,    23,
      43,    50,    51,    50,    19,    27,    29,    31,    18,    40,
      25,    42,    51,    56,    54,    18,    36,    43,    43,    43,
      51,    28,    34,    30,    43,    45,    46,    48,    48,    19,
      24,    24,    43,    24,    13,    28,    52,    53,    18,    34,
      34,    34,    24,    43,    35,    44,    34,    33,    28,    30,
      19,    37,    43,    43,    43,    20,    43,    49,    33,    33,
      33,    44,    25,    43,    37
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
#define YYERROR		goto yyerrlab1


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
# define YYLLOC_DEFAULT(Current, Rhs, N)         \
  Current.first_line   = Rhs[1].first_line;      \
  Current.first_column = Rhs[1].first_column;    \
  Current.last_line    = Rhs[N].last_line;       \
  Current.last_column  = Rhs[N].last_column;
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
| TOP (cinluded).                                                   |
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
  unsigned int yylineno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylineno);
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

#if YYMAXDEPTH == 0
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
#line 223 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Rec = Records.getClass(*yyvsp[0].StrVal);
    if (yyval.Rec == 0) {
      err() << "Couldn't find class '" << *yyvsp[0].StrVal << "'!\n";
      exit(1);
    }
    delete yyvsp[0].StrVal;
  ;}
    break;

  case 3:
#line 234 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {                       // string type
    yyval.Ty = new StringRecTy();
  ;}
    break;

  case 4:
#line 236 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {                           // bit type
    yyval.Ty = new BitRecTy();
  ;}
    break;

  case 5:
#line 238 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {           // bits<x> type
    yyval.Ty = new BitsRecTy(yyvsp[-1].IntVal);
  ;}
    break;

  case 6:
#line 240 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {                           // int type
    yyval.Ty = new IntRecTy();
  ;}
    break;

  case 7:
#line 242 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {          // list<x> type
    yyval.Ty = new ListRecTy(yyvsp[-1].Ty);
  ;}
    break;

  case 8:
#line 244 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {                          // code type
    yyval.Ty = new CodeRecTy();
  ;}
    break;

  case 9:
#line 246 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {                           // dag type
    yyval.Ty = new DagRecTy();
  ;}
    break;

  case 10:
#line 248 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {                       // Record Type
    yyval.Ty = new RecordRecTy(yyvsp[0].Rec);
  ;}
    break;

  case 11:
#line 252 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.IntVal = 0; ;}
    break;

  case 12:
#line 252 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.IntVal = 1; ;}
    break;

  case 13:
#line 254 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.Initializer = 0; ;}
    break;

  case 14:
#line 254 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.Initializer = yyvsp[0].Initializer; ;}
    break;

  case 15:
#line 256 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = new IntInit(yyvsp[0].IntVal);
  ;}
    break;

  case 16:
#line 258 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = new StringInit(*yyvsp[0].StrVal);
    delete yyvsp[0].StrVal;
  ;}
    break;

  case 17:
#line 261 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = new CodeInit(*yyvsp[0].StrVal);
    delete yyvsp[0].StrVal;
  ;}
    break;

  case 18:
#line 264 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = new UnsetInit();
  ;}
    break;

  case 19:
#line 266 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    BitsInit *Init = new BitsInit(yyvsp[-1].FieldList->size());
    for (unsigned i = 0, e = yyvsp[-1].FieldList->size(); i != e; ++i) {
      struct Init *Bit = (*yyvsp[-1].FieldList)[i]->convertInitializerTo(new BitRecTy());
      if (Bit == 0) {
        err() << "Element #" << i << " (" << *(*yyvsp[-1].FieldList)[i]
       	      << ") is not convertable to a bit!\n";
        exit(1);
      }
      Init->setBit(yyvsp[-1].FieldList->size()-i-1, Bit);
    }
    yyval.Initializer = Init;
    delete yyvsp[-1].FieldList;
  ;}
    break;

  case 20:
#line 279 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    // This is a CLASS<initvalslist> expression.  This is supposed to synthesize
    // a new anonymous definition, deriving from CLASS<initvalslist> with no
    // body.
    Record *Class = Records.getClass(*yyvsp[-3].StrVal);
    if (!Class) {
      err() << "Expected a class, got '" << *yyvsp[-3].StrVal << "'!\n";
      exit(1);
    }
    delete yyvsp[-3].StrVal;
    
    static unsigned AnonCounter = 0;
    Record *OldRec = CurRec;  // Save CurRec.
    
    // Create the new record, set it as CurRec temporarily.
    CurRec = new Record("anonymous.val."+utostr(AnonCounter++));
    addSubClass(Class, *yyvsp[-1].FieldList);    // Add info about the subclass to CurRec.
    delete yyvsp[-1].FieldList;  // Free up the template args.
    
    CurRec->resolveReferences();
    
    Records.addDef(CurRec);
    
    // The result of the expression is a reference to the new record.
    yyval.Initializer = new DefInit(CurRec);
    
    // Restore the old CurRec
    CurRec = OldRec;
  ;}
    break;

  case 21:
#line 307 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    if (const RecordVal *RV = (CurRec ? CurRec->getValue(*yyvsp[0].StrVal) : 0)) {
      yyval.Initializer = new VarInit(*yyvsp[0].StrVal, RV->getType());
    } else if (CurRec && CurRec->isTemplateArg(CurRec->getName()+":"+*yyvsp[0].StrVal)) {
      const RecordVal *RV = CurRec->getValue(CurRec->getName()+":"+*yyvsp[0].StrVal);
      assert(RV && "Template arg doesn't exist??");
      yyval.Initializer = new VarInit(CurRec->getName()+":"+*yyvsp[0].StrVal, RV->getType());
    } else if (Record *D = Records.getDef(*yyvsp[0].StrVal)) {
      yyval.Initializer = new DefInit(D);
    } else {
      err() << "Variable not defined: '" << *yyvsp[0].StrVal << "'!\n";
      exit(1);
    }
    
    delete yyvsp[0].StrVal;
  ;}
    break;

  case 22:
#line 322 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = yyvsp[-3].Initializer->convertInitializerBitRange(*yyvsp[-1].BitList);
    if (yyval.Initializer == 0) {
      err() << "Invalid bit range for value '" << *yyvsp[-3].Initializer << "'!\n";
      exit(1);
    }
    delete yyvsp[-1].BitList;
  ;}
    break;

  case 23:
#line 329 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = new ListInit(*yyvsp[-1].FieldList);
    delete yyvsp[-1].FieldList;
  ;}
    break;

  case 24:
#line 332 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    if (!yyvsp[-2].Initializer->getFieldType(*yyvsp[0].StrVal)) {
      err() << "Cannot access field '" << *yyvsp[0].StrVal << "' of value '" << *yyvsp[-2].Initializer << "!\n";
      exit(1);
    }
    yyval.Initializer = new FieldInit(yyvsp[-2].Initializer, *yyvsp[0].StrVal);
    delete yyvsp[0].StrVal;
  ;}
    break;

  case 25:
#line 339 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    Record *D = Records.getDef(*yyvsp[-2].StrVal);
    if (D == 0) {
      err() << "Invalid def '" << *yyvsp[-2].StrVal << "'!\n";
      exit(1);
    }
    yyval.Initializer = new DagInit(D, *yyvsp[-1].DagValueList);
    delete yyvsp[-2].StrVal; delete yyvsp[-1].DagValueList;
  ;}
    break;

  case 26:
#line 347 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    std::reverse(yyvsp[-1].BitList->begin(), yyvsp[-1].BitList->end());
    yyval.Initializer = yyvsp[-3].Initializer->convertInitListSlice(*yyvsp[-1].BitList);
    if (yyval.Initializer == 0) {
      err() << "Invalid list slice for value '" << *yyvsp[-3].Initializer << "'!\n";
      exit(1);
    }
    delete yyvsp[-1].BitList;
  ;}
    break;

  case 27:
#line 355 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = yyvsp[-3].Initializer->getBinaryOp(Init::SHL, yyvsp[-1].Initializer);
    if (yyval.Initializer == 0) {
      err() << "Cannot shift values '" << *yyvsp[-3].Initializer << "' and '" << *yyvsp[-1].Initializer << "'!\n";
      exit(1);
    }
  ;}
    break;

  case 28:
#line 361 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = yyvsp[-3].Initializer->getBinaryOp(Init::SRA, yyvsp[-1].Initializer);
    if (yyval.Initializer == 0) {
      err() << "Cannot shift values '" << *yyvsp[-3].Initializer << "' and '" << *yyvsp[-1].Initializer << "'!\n";
      exit(1);
    }
  ;}
    break;

  case 29:
#line 367 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.Initializer = yyvsp[-3].Initializer->getBinaryOp(Init::SRL, yyvsp[-1].Initializer);
    if (yyval.Initializer == 0) {
      err() << "Cannot shift values '" << *yyvsp[-3].Initializer << "' and '" << *yyvsp[-1].Initializer << "'!\n";
      exit(1);
    }
  ;}
    break;

  case 30:
#line 375 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.StrVal = new std::string();
  ;}
    break;

  case 31:
#line 378 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.StrVal = yyvsp[0].StrVal;
  ;}
    break;

  case 32:
#line 382 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.DagValueList = new std::vector<std::pair<Init*, std::string> >();
    yyval.DagValueList->push_back(std::make_pair(yyvsp[-1].Initializer, *yyvsp[0].StrVal));
    delete yyvsp[0].StrVal;
  ;}
    break;

  case 33:
#line 387 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyvsp[-3].DagValueList->push_back(std::make_pair(yyvsp[-1].Initializer, *yyvsp[0].StrVal));
    delete yyvsp[0].StrVal;
    yyval.DagValueList = yyvsp[-3].DagValueList;
  ;}
    break;

  case 34:
#line 393 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.DagValueList = new std::vector<std::pair<Init*, std::string> >();
  ;}
    break;

  case 35:
#line 396 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.DagValueList = yyvsp[0].DagValueList; ;}
    break;

  case 36:
#line 399 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.BitList = new std::vector<unsigned>();
    yyval.BitList->push_back(yyvsp[0].IntVal);
  ;}
    break;

  case 37:
#line 402 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    if (yyvsp[-2].IntVal < 0 || yyvsp[0].IntVal < 0) {
      err() << "Invalid range: " << yyvsp[-2].IntVal << "-" << yyvsp[0].IntVal << "!\n";
      exit(1);
    }
    yyval.BitList = new std::vector<unsigned>();
    if (yyvsp[-2].IntVal < yyvsp[0].IntVal) {
      for (int i = yyvsp[-2].IntVal; i <= yyvsp[0].IntVal; ++i)
        yyval.BitList->push_back(i);
    } else {
      for (int i = yyvsp[-2].IntVal; i >= yyvsp[0].IntVal; --i)
        yyval.BitList->push_back(i);
    }
  ;}
    break;

  case 38:
#line 415 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyvsp[0].IntVal = -yyvsp[0].IntVal;
    if (yyvsp[-1].IntVal < 0 || yyvsp[0].IntVal < 0) {
      err() << "Invalid range: " << yyvsp[-1].IntVal << "-" << yyvsp[0].IntVal << "!\n";
      exit(1);
    }
    yyval.BitList = new std::vector<unsigned>();
    if (yyvsp[-1].IntVal < yyvsp[0].IntVal) {
      for (int i = yyvsp[-1].IntVal; i <= yyvsp[0].IntVal; ++i)
        yyval.BitList->push_back(i);
    } else {
      for (int i = yyvsp[-1].IntVal; i >= yyvsp[0].IntVal; --i)
        yyval.BitList->push_back(i);
    }
  ;}
    break;

  case 39:
#line 429 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    (yyval.BitList=yyvsp[-2].BitList)->push_back(yyvsp[0].IntVal);
  ;}
    break;

  case 40:
#line 431 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    if (yyvsp[-2].IntVal < 0 || yyvsp[0].IntVal < 0) {
      err() << "Invalid range: " << yyvsp[-2].IntVal << "-" << yyvsp[0].IntVal << "!\n";
      exit(1);
    }
    yyval.BitList = yyvsp[-4].BitList;
    if (yyvsp[-2].IntVal < yyvsp[0].IntVal) {
      for (int i = yyvsp[-2].IntVal; i <= yyvsp[0].IntVal; ++i)
        yyval.BitList->push_back(i);
    } else {
      for (int i = yyvsp[-2].IntVal; i >= yyvsp[0].IntVal; --i)
        yyval.BitList->push_back(i);
    }
  ;}
    break;

  case 41:
#line 444 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyvsp[0].IntVal = -yyvsp[0].IntVal;
    if (yyvsp[-1].IntVal < 0 || yyvsp[0].IntVal < 0) {
      err() << "Invalid range: " << yyvsp[-1].IntVal << "-" << yyvsp[0].IntVal << "!\n";
      exit(1);
    }
    yyval.BitList = yyvsp[-3].BitList;
    if (yyvsp[-1].IntVal < yyvsp[0].IntVal) {
      for (int i = yyvsp[-1].IntVal; i <= yyvsp[0].IntVal; ++i)
        yyval.BitList->push_back(i);
    } else {
      for (int i = yyvsp[-1].IntVal; i >= yyvsp[0].IntVal; --i)
        yyval.BitList->push_back(i);
    }
  ;}
    break;

  case 42:
#line 460 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.BitList = yyvsp[0].BitList; std::reverse(yyvsp[0].BitList->begin(), yyvsp[0].BitList->end()); ;}
    break;

  case 43:
#line 462 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.BitList = 0; ;}
    break;

  case 44:
#line 462 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.BitList = yyvsp[-1].BitList; ;}
    break;

  case 45:
#line 466 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.FieldList = new std::vector<Init*>();
  ;}
    break;

  case 46:
#line 468 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.FieldList = yyvsp[0].FieldList;
  ;}
    break;

  case 47:
#line 472 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.FieldList = new std::vector<Init*>();
    yyval.FieldList->push_back(yyvsp[0].Initializer);
  ;}
    break;

  case 48:
#line 475 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    (yyval.FieldList = yyvsp[-2].FieldList)->push_back(yyvsp[0].Initializer);
  ;}
    break;

  case 49:
#line 479 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  std::string DecName = *yyvsp[-1].StrVal;
  if (ParsingTemplateArgs)
    DecName = CurRec->getName() + ":" + DecName;

  addValue(RecordVal(DecName, yyvsp[-2].Ty, yyvsp[-3].IntVal));
  setValue(DecName, 0, yyvsp[0].Initializer);
  yyval.StrVal = new std::string(DecName);
;}
    break;

  case 50:
#line 489 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  delete yyvsp[-1].StrVal;
;}
    break;

  case 51:
#line 491 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  setValue(*yyvsp[-4].StrVal, yyvsp[-3].BitList, yyvsp[-1].Initializer);
  delete yyvsp[-4].StrVal;
  delete yyvsp[-3].BitList;
;}
    break;

  case 56:
#line 500 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.SubClassRef = new SubClassRefTy(yyvsp[0].Rec, new std::vector<Init*>());
  ;}
    break;

  case 57:
#line 502 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.SubClassRef = new SubClassRefTy(yyvsp[-3].Rec, yyvsp[-1].FieldList);
  ;}
    break;

  case 58:
#line 506 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.SubClassList = new std::vector<SubClassRefTy>();
    yyval.SubClassList->push_back(*yyvsp[0].SubClassRef);
    delete yyvsp[0].SubClassRef;
  ;}
    break;

  case 59:
#line 511 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    (yyval.SubClassList=yyvsp[-2].SubClassList)->push_back(*yyvsp[0].SubClassRef);
    delete yyvsp[0].SubClassRef;
  ;}
    break;

  case 60:
#line 516 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.SubClassList = new std::vector<SubClassRefTy>();
  ;}
    break;

  case 61:
#line 519 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    yyval.SubClassList = yyvsp[0].SubClassList;
  ;}
    break;

  case 62:
#line 523 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  CurRec->addTemplateArg(*yyvsp[0].StrVal);
  delete yyvsp[0].StrVal;
;}
    break;

  case 63:
#line 526 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  CurRec->addTemplateArg(*yyvsp[0].StrVal);
  delete yyvsp[0].StrVal;
;}
    break;

  case 64:
#line 531 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {;}
    break;

  case 67:
#line 534 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.StrVal = yyvsp[0].StrVal; ;}
    break;

  case 68:
#line 534 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { yyval.StrVal = new std::string(); ;}
    break;

  case 69:
#line 536 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
           static unsigned AnonCounter = 0;
           if (yyvsp[0].StrVal->empty())
             *yyvsp[0].StrVal = "anonymous."+utostr(AnonCounter++);
           CurRec = new Record(*yyvsp[0].StrVal);
           delete yyvsp[0].StrVal;
           ParsingTemplateArgs = true;
         ;}
    break;

  case 70:
#line 543 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
           ParsingTemplateArgs = false;
           for (unsigned i = 0, e = yyvsp[0].SubClassList->size(); i != e; ++i) {
             addSubClass((*yyvsp[0].SubClassList)[i].first, *(*yyvsp[0].SubClassList)[i].second);
             // Delete the template arg values for the class
             delete (*yyvsp[0].SubClassList)[i].second;
           }
           delete yyvsp[0].SubClassList;   // Delete the class list...

           // Process any variables on the set stack...
           for (unsigned i = 0, e = LetStack.size(); i != e; ++i)
             for (unsigned j = 0, e = LetStack[i].size(); j != e; ++j)
               setValue(LetStack[i][j].Name,
                        LetStack[i][j].HasBits ? &LetStack[i][j].Bits : 0,
                        LetStack[i][j].Value);
         ;}
    break;

  case 71:
#line 558 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
           yyval.Rec = CurRec;
           CurRec = 0;
         ;}
    break;

  case 72:
#line 563 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  if (Records.getClass(yyvsp[0].Rec->getName())) {
    err() << "Class '" << yyvsp[0].Rec->getName() << "' already defined!\n";
    exit(1);
  }
  Records.addClass(yyval.Rec = yyvsp[0].Rec);
;}
    break;

  case 73:
#line 571 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  yyvsp[0].Rec->resolveReferences();

  // If ObjectBody has template arguments, it's an error.
  if (!yyvsp[0].Rec->getTemplateArgs().empty()) {
    err() << "Def '" << yyvsp[0].Rec->getName()
          << "' is not permitted to have template arguments!\n";
    exit(1);
  }
  // Ensure redefinition doesn't happen.
  if (Records.getDef(yyvsp[0].Rec->getName())) {
    err() << "Def '" << yyvsp[0].Rec->getName() << "' already defined!\n";
    exit(1);
  }
  Records.addDef(yyval.Rec = yyvsp[0].Rec);
;}
    break;

  case 76:
#line 591 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
  LetStack.back().push_back(LetRecord(*yyvsp[-3].StrVal, yyvsp[-2].BitList, yyvsp[0].Initializer));
  delete yyvsp[-3].StrVal; delete yyvsp[-2].BitList;
;}
    break;

  case 79:
#line 599 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    { LetStack.push_back(std::vector<LetRecord>()); ;}
    break;

  case 81:
#line 602 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    LetStack.pop_back();
  ;}
    break;

  case 82:
#line 605 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {
    LetStack.pop_back();
  ;}
    break;

  case 83:
#line 609 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {;}
    break;

  case 84:
#line 609 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {;}
    break;

  case 85:
#line 611 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"
    {;}
    break;


    }

/* Line 999 of yacc.c.  */
#line 2018 "FileParser.tab.c"

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
	  char *yymsg;
	  int yyx, yycount;

	  yycount = 0;
	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  for (yyx = yyn < 0 ? -yyn : 0;
	       yyx < (int) (sizeof (yytname) / sizeof (char *)); yyx++)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      yysize += yystrlen (yytname[yyx]) + 15, yycount++;
	  yysize += yystrlen ("syntax error, unexpected ") + 1;
	  yysize += yystrlen (yytname[yytype]);
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yycount = 0;
		  for (yyx = yyn < 0 ? -yyn : 0;
		       yyx < (int) (sizeof (yytname) / sizeof (char *));
		       yyx++)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			const char *yyq = ! yycount ? ", expecting " : " or ";
			yyp = yystpcpy (yyp, yyq);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yycount++;
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

      /* Return failure if at end of input.  */
      if (yychar == YYEOF)
        {
	  /* Pop the error token.  */
          YYPOPSTACK;
	  /* Pop the rest of the stack.  */
	  while (yyss < yyssp)
	    {
	      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
	      yydestruct (yystos[*yyssp], yyvsp);
	      YYPOPSTACK;
	    }
	  YYABORT;
        }

      YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
      yydestruct (yytoken, &yylval);
      yychar = YYEMPTY;

    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*----------------------------------------------------.
| yyerrlab1 -- error raised explicitly by an action.  |
`----------------------------------------------------*/
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
      yyvsp--;
      yystate = *--yyssp;

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


#line 613 "/home/vadve/lattner/llvm/utils/TableGen/FileParser.y"


int yyerror(const char *ErrorMsg) {
  err() << "Error parsing: " << ErrorMsg << "\n";
  exit(1);
}

