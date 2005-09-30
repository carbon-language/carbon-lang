
/*  A Bison parser, made from /Users/sabre/llvm/utils/TableGen/FileParser.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define yyparse Fileparse
#define yylex Filelex
#define yyerror Fileerror
#define yylval Filelval
#define yychar Filechar
#define yydebug Filedebug
#define yynerrs Filenerrs
#define	INT	257
#define	BIT	258
#define	STRING	259
#define	BITS	260
#define	LIST	261
#define	CODE	262
#define	DAG	263
#define	CLASS	264
#define	DEF	265
#define	FIELD	266
#define	LET	267
#define	IN	268
#define	SHLTOK	269
#define	SRATOK	270
#define	SRLTOK	271
#define	INTVAL	272
#define	ID	273
#define	VARNAME	274
#define	STRVAL	275
#define	CODEFRAGMENT	276

#line 14 "/Users/sabre/llvm/utils/TableGen/FileParser.y"

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


#line 189 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
typedef union {
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
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		160
#define	YYFLAG		-32768
#define	YYNTBASE	38

#define YYTRANSLATE(x) ((unsigned)(x) <= 276 ? yytranslate[x] : 78)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    32,
    33,     2,     2,    34,    36,    31,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    35,    37,    23,
    25,    24,    26,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    29,     2,    30,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    27,     2,    28,     2,     2,     2,     2,     2,
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
     2,     2,     2,     2,     2,     1,     3,     4,     5,     6,
     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    17,    18,    19,    20,    21,    22
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     4,     6,    11,    13,    18,    20,    22,    24,
    25,    27,    28,    31,    33,    35,    37,    39,    43,    48,
    50,    55,    59,    63,    68,    73,    80,    87,    94,    95,
    98,   101,   106,   107,   109,   111,   115,   118,   122,   128,
   133,   135,   136,   140,   141,   143,   145,   149,   154,   157,
   164,   165,   168,   170,   174,   176,   181,   183,   187,   188,
   191,   193,   197,   201,   202,   204,   206,   207,   209,   211,
   213,   214,   218,   219,   220,   227,   231,   233,   235,   240,
   242,   246,   247,   252,   257,   260,   262,   265
};

static const short yyrhs[] = {    19,
     0,     5,     0,     4,     0,     6,    23,    18,    24,     0,
     3,     0,     7,    23,    39,    24,     0,     8,     0,     9,
     0,    38,     0,     0,    12,     0,     0,    25,    42,     0,
    18,     0,    21,     0,    22,     0,    26,     0,    27,    49,
    28,     0,    19,    23,    50,    24,     0,    19,     0,    42,
    27,    47,    28,     0,    29,    49,    30,     0,    42,    31,
    19,     0,    32,    19,    45,    33,     0,    42,    29,    47,
    30,     0,    15,    32,    42,    34,    42,    33,     0,    16,
    32,    42,    34,    42,    33,     0,    17,    32,    42,    34,
    42,    33,     0,     0,    35,    20,     0,    42,    43,     0,
    44,    34,    42,    43,     0,     0,    44,     0,    18,     0,
    18,    36,    18,     0,    18,    18,     0,    46,    34,    18,
     0,    46,    34,    18,    36,    18,     0,    46,    34,    18,
    18,     0,    46,     0,     0,    27,    47,    28,     0,     0,
    50,     0,    42,     0,    50,    34,    42,     0,    40,    39,
    19,    41,     0,    51,    37,     0,    13,    19,    48,    25,
    42,    37,     0,     0,    53,    52,     0,    37,     0,    27,
    53,    28,     0,    38,     0,    38,    23,    50,    24,     0,
    55,     0,    56,    34,    55,     0,     0,    35,    56,     0,
    51,     0,    58,    34,    51,     0,    23,    58,    24,     0,
     0,    59,     0,    19,     0,     0,    61,     0,    62,     0,
    62,     0,     0,    57,    66,    54,     0,     0,     0,    10,
    63,    68,    60,    69,    65,     0,    11,    64,    65,     0,
    67,     0,    70,     0,    19,    48,    25,    42,     0,    72,
     0,    73,    34,    72,     0,     0,    13,    75,    73,    14,
     0,    74,    27,    76,    28,     0,    74,    71,     0,    71,
     0,    76,    71,     0,    76,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   223,   234,   236,   238,   240,   242,   244,   246,   248,   252,
   252,   254,   254,   256,   258,   261,   264,   266,   279,   307,
   322,   329,   332,   339,   347,   355,   361,   367,   375,   378,
   382,   387,   393,   396,   399,   402,   415,   429,   431,   444,
   460,   462,   462,   466,   468,   472,   475,   479,   489,   491,
   497,   497,   498,   498,   500,   502,   506,   511,   516,   519,
   523,   526,   531,   532,   532,   534,   534,   536,   543,   561,
   573,   587,   592,   594,   596,   600,   609,   609,   611,   616,
   616,   619,   619,   622,   625,   629,   629,   631
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","INT","BIT",
"STRING","BITS","LIST","CODE","DAG","CLASS","DEF","FIELD","LET","IN","SHLTOK",
"SRATOK","SRLTOK","INTVAL","ID","VARNAME","STRVAL","CODEFRAGMENT","'<'","'>'",
"'='","'?'","'{'","'}'","'['","']'","'.'","'('","')'","','","':'","'-'","';'",
"ClassID","Type","OptPrefix","OptValue","Value","OptVarName","DagArgListNE",
"DagArgList","RBitList","BitList","OptBitList","ValueList","ValueListNE","Declaration",
"BodyItem","BodyList","Body","SubClassRef","ClassListNE","ClassList","DeclListNE",
"TemplateArgList","OptTemplateArgList","OptID","ObjectName","ClassName","DefName",
"ObjectBody","@1","ClassInst","@2","@3","DefInst","Object","LETItem","LETList",
"LETCommand","@4","ObjectList","File", NULL
};
#endif

static const short yyr1[] = {     0,
    38,    39,    39,    39,    39,    39,    39,    39,    39,    40,
    40,    41,    41,    42,    42,    42,    42,    42,    42,    42,
    42,    42,    42,    42,    42,    42,    42,    42,    43,    43,
    44,    44,    45,    45,    46,    46,    46,    46,    46,    46,
    47,    48,    48,    49,    49,    50,    50,    51,    52,    52,
    53,    53,    54,    54,    55,    55,    56,    56,    57,    57,
    58,    58,    59,    60,    60,    61,    61,    62,    63,    64,
    66,    65,    68,    69,    67,    70,    71,    71,    72,    73,
    73,    75,    74,    71,    71,    76,    76,    77
};

static const short yyr2[] = {     0,
     1,     1,     1,     4,     1,     4,     1,     1,     1,     0,
     1,     0,     2,     1,     1,     1,     1,     3,     4,     1,
     4,     3,     3,     4,     4,     6,     6,     6,     0,     2,
     2,     4,     0,     1,     1,     3,     2,     3,     5,     4,
     1,     0,     3,     0,     1,     1,     3,     4,     2,     6,
     0,     2,     1,     3,     1,     4,     1,     3,     0,     2,
     1,     3,     3,     0,     1,     1,     0,     1,     1,     1,
     0,     3,     0,     0,     6,     3,     1,     1,     4,     1,
     3,     0,     4,     4,     2,     1,     2,     1
};

static const short yydefact[] = {     0,
    67,    67,    82,    77,    78,    86,     0,    88,    66,    68,
    69,    73,    70,    59,     0,     0,    85,    87,    64,     0,
    71,    76,    42,    80,     0,     0,    10,    65,    74,     1,
    55,    57,    60,     0,     0,     0,    83,     0,    84,    11,
     0,    61,     0,    59,     0,     0,    51,    53,    72,    35,
    41,     0,     0,    81,     5,     3,     2,     0,     0,     7,
     8,     9,     0,    63,    10,    75,     0,     0,     0,    14,
    20,    15,    16,    17,    44,    44,     0,    46,     0,    58,
    10,    37,     0,     0,    43,    79,     0,     0,    12,    62,
     0,     0,     0,     0,     0,    45,     0,    33,     0,     0,
     0,    56,     0,     0,    54,     0,    52,    36,    38,     0,
     0,     0,    48,     0,     0,     0,     0,    18,    22,    29,
    34,     0,     0,     0,    23,    47,    42,    49,    40,     0,
     4,     6,    13,     0,     0,     0,    19,     0,    31,     0,
    24,    21,    25,     0,    39,     0,     0,     0,    30,    29,
     0,    26,    27,    28,    32,     0,    50,     0,     0,     0
};

static const short yydefgoto[] = {    31,
    63,    41,   113,    78,   139,   121,   122,    51,    52,    36,
    95,    96,    42,   107,    81,    49,    32,    33,    21,    43,
    28,    29,    10,    11,    12,    14,    22,    34,     4,    19,
    44,     5,     6,    24,    25,     7,    15,     8,   158
};

static const short yypact[] = {    56,
    -4,    -4,-32768,-32768,-32768,-32768,     1,    56,-32768,-32768,
-32768,-32768,-32768,     8,     2,    56,-32768,-32768,    23,    30,
-32768,-32768,    50,-32768,   -11,    -3,    66,-32768,-32768,-32768,
    62,-32768,    55,    34,    61,    68,-32768,     2,-32768,-32768,
    49,-32768,    -8,     8,    15,    30,-32768,-32768,-32768,   -14,
    82,    67,    15,-32768,-32768,-32768,-32768,    95,    97,-32768,
-32768,-32768,    77,-32768,    66,-32768,    89,    90,    91,-32768,
   101,-32768,-32768,-32768,    15,    15,   106,    73,    39,-32768,
     7,-32768,   108,   109,-32768,    73,   110,    49,   104,-32768,
    15,    15,    15,    15,   102,    98,   103,    15,    61,    61,
   112,-32768,    15,   115,-32768,    99,-32768,-32768,    -9,   111,
   113,    15,-32768,    57,    63,    72,    41,-32768,-32768,    45,
   105,   107,   114,   116,-32768,    73,    50,-32768,-32768,   120,
-32768,-32768,    73,    15,    15,    15,-32768,   121,-32768,    15,
-32768,-32768,-32768,   118,-32768,    78,    81,    86,-32768,    45,
    15,-32768,-32768,-32768,-32768,    33,-32768,   144,   145,-32768
};

static const short yypgoto[] = {   -40,
    59,-32768,-32768,   -53,    -1,-32768,-32768,-32768,   -82,    21,
    74,   -43,   -52,-32768,-32768,-32768,   117,-32768,-32768,-32768,
-32768,-32768,-32768,   149,-32768,-32768,   122,-32768,-32768,-32768,
-32768,-32768,    -2,   119,-32768,-32768,-32768,   136,-32768
};


#define	YYLAST		166


static const short yytable[] = {    86,
    62,    79,    37,    82,    17,    18,     1,     2,   129,     3,
     1,     2,    90,     3,     9,    64,   123,   124,    40,   104,
    23,    83,    38,    18,    39,    65,   130,    16,   106,    67,
    68,    69,    70,    71,   105,    72,    73,   114,   115,   116,
    74,    75,    20,    76,   120,    27,    77,    62,    30,   126,
   117,    55,    56,    57,    58,    59,    60,    61,   133,    99,
    47,   100,   102,   101,   137,     1,     2,    30,     3,   157,
    48,    99,   103,   100,   103,   101,    35,    40,    50,   138,
   146,   147,   148,    99,    45,   100,   150,   101,    46,    99,
   134,   100,    53,   101,    85,    89,   135,   156,    99,    99,
   100,   100,   101,   101,    99,   136,   100,    99,   101,   100,
   152,   101,    99,   153,   100,    84,   101,    87,   154,    88,
    91,    92,    93,    94,    98,   108,   109,   110,   112,   118,
   125,   103,   119,   127,   131,   128,   132,   145,   140,   141,
   149,   142,   151,   159,   160,   143,   111,   144,   155,    97,
    13,    26,     0,     0,     0,     0,    54,     0,     0,     0,
     0,     0,    80,     0,     0,    66
};

static const short yycheck[] = {    53,
    41,    45,    14,    18,     7,     8,    10,    11,    18,    13,
    10,    11,    65,    13,    19,    24,    99,   100,    12,    13,
    19,    36,    34,    26,    28,    34,    36,    27,    81,    15,
    16,    17,    18,    19,    28,    21,    22,    91,    92,    93,
    26,    27,    35,    29,    98,    23,    32,    88,    19,   103,
    94,     3,     4,     5,     6,     7,     8,     9,   112,    27,
    27,    29,    24,    31,    24,    10,    11,    19,    13,    37,
    37,    27,    34,    29,    34,    31,    27,    12,    18,    35,
   134,   135,   136,    27,    23,    29,   140,    31,    34,    27,
    34,    29,    25,    31,    28,    19,    34,   151,    27,    27,
    29,    29,    31,    31,    27,    34,    29,    27,    31,    29,
    33,    31,    27,    33,    29,    34,    31,    23,    33,    23,
    32,    32,    32,    23,    19,    18,    18,    18,    25,    28,
    19,    34,    30,    19,    24,    37,    24,    18,    34,    33,
    20,    28,    25,     0,     0,    30,    88,   127,   150,    76,
     2,    16,    -1,    -1,    -1,    -1,    38,    -1,    -1,    -1,
    -1,    -1,    46,    -1,    -1,    44
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/share/bison.simple"
/* This file comes from bison-1.28.  */

/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Free Software Foundation, Inc.

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

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

#ifndef YYSTACK_USE_ALLOCA
#ifdef alloca
#define YYSTACK_USE_ALLOCA
#else /* alloca not defined */
#ifdef __GNUC__
#define YYSTACK_USE_ALLOCA
#define alloca __builtin_alloca
#else /* not GNU C.  */
#if (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc) || defined (__sgi) || (defined (__sun) && defined (__i386))
#define YYSTACK_USE_ALLOCA
#include <alloca.h>
#else /* not sparc */
/* We think this test detects Watcom and Microsoft C.  */
/* This used to test MSDOS, but that is a bad idea
   since that symbol is in the user namespace.  */
#if (defined (_MSDOS) || defined (_MSDOS_)) && !defined (__TURBOC__)
#if 0 /* No need for malloc.h, which pollutes the namespace;
	 instead, just don't use alloca.  */
#include <malloc.h>
#endif
#else /* not MSDOS, or __TURBOC__ */
#if defined(_AIX)
/* I don't know what this was needed for, but it pollutes the namespace.
   So I turned it off.   rms, 2 May 1997.  */
/* #include <malloc.h>  */
 #pragma alloca
#define YYSTACK_USE_ALLOCA
#else /* not MSDOS, or __TURBOC__, or _AIX */
#if 0
#ifdef __hpux /* haible@ilog.fr says this works for HPUX 9.05 and up,
		 and on HPUX 10.  Eventually we can turn this on.  */
#define YYSTACK_USE_ALLOCA
#define alloca __builtin_alloca
#endif /* __hpux */
#endif
#endif /* not _AIX */
#endif /* not MSDOS, or __TURBOC__ */
#endif /* not sparc */
#endif /* not GNU C */
#endif /* alloca not defined */
#endif /* YYSTACK_USE_ALLOCA not defined */

#ifdef YYSTACK_USE_ALLOCA
#define YYSTACK_ALLOC alloca
#else
#define YYSTACK_ALLOC malloc
#endif

/* Note: there must be only one dollar sign in this file.
   It is replaced by the list of actions, each action
   as one case of the switch.  */

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		-2
#define YYEOF		0
#define YYACCEPT	goto yyacceptlab
#define YYABORT 	goto yyabortlab
#define YYERROR		goto yyerrlab1
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL		goto yyerrlab
#define YYRECOVERING()  (!!yyerrstatus)
#define YYBACKUP(token, value) \
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    { yychar = (token), yylval = (value);			\
      yychar1 = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { yyerror ("syntax error: cannot back up"); YYERROR; }	\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

#ifndef YYPURE
#define YYLEX		yylex()
#endif

#ifdef YYPURE
#ifdef YYLSP_NEEDED
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, &yylloc, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval, &yylloc)
#endif
#else /* not YYLSP_NEEDED */
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval)
#endif
#endif /* not YYLSP_NEEDED */
#endif

/* If nonreentrant, generate the variables here */

#ifndef YYPURE

int	yychar;			/*  the lookahead symbol		*/
YYSTYPE	yylval;			/*  the semantic value of the		*/
				/*  lookahead symbol			*/

#ifdef YYLSP_NEEDED
YYLTYPE yylloc;			/*  location data for the lookahead	*/
				/*  symbol				*/
#endif

int yynerrs;			/*  number of parse errors so far       */
#endif  /* not YYPURE */

#if YYDEBUG != 0
int yydebug;			/*  nonzero means print parse trace	*/
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif

/*  YYINITDEPTH indicates the initial size of the parser's stacks	*/

#ifndef	YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

/* Define __yy_memcpy.  Note that the size argument
   should be passed with type unsigned int, because that is what the non-GCC
   definitions require.  With GCC, __builtin_memcpy takes an arg
   of type size_t, but it can handle unsigned int.  */

#if __GNUC__ > 1		/* GNU C and GNU C++ define this.  */
#define __yy_memcpy(TO,FROM,COUNT)	__builtin_memcpy(TO,FROM,COUNT)
#else				/* not GNU C or C++ */
#ifndef __cplusplus

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (to, from, count)
     char *to;
     char *from;
     unsigned int count;
{
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#else /* __cplusplus */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (char *to, char *from, unsigned int count)
{
  register char *t = to;
  register char *f = from;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#endif
#endif

#line 217 "/usr/share/bison.simple"

/* The user can define YYPARSE_PARAM as the name of an argument to be passed
   into yyparse.  The argument should have type void *.
   It should actually point to an object.
   Grammar actions can access the variable by casting it
   to the proper pointer type.  */

#ifdef YYPARSE_PARAM
#ifdef __cplusplus
#define YYPARSE_PARAM_ARG void *YYPARSE_PARAM
#define YYPARSE_PARAM_DECL
#else /* not __cplusplus */
#define YYPARSE_PARAM_ARG YYPARSE_PARAM
#define YYPARSE_PARAM_DECL void *YYPARSE_PARAM;
#endif /* not __cplusplus */
#else /* not YYPARSE_PARAM */
#define YYPARSE_PARAM_ARG
#define YYPARSE_PARAM_DECL
#endif /* not YYPARSE_PARAM */

/* Prevent warning if -Wstrict-prototypes.  */
#ifdef __GNUC__
#ifdef YYPARSE_PARAM
int yyparse (void *);
#else
int yyparse (void);
#endif
#endif

int
yyparse(YYPARSE_PARAM_ARG)
     YYPARSE_PARAM_DECL
{
  register int yystate;
  register int yyn;
  register short *yyssp;
  register YYSTYPE *yyvsp;
  int yyerrstatus;	/*  number of tokens to shift before error messages enabled */
  int yychar1 = 0;		/*  lookahead token as an internal (translated) token number */

  short	yyssa[YYINITDEPTH];	/*  the state stack			*/
  YYSTYPE yyvsa[YYINITDEPTH];	/*  the semantic value stack		*/

  short *yyss = yyssa;		/*  refer to the stacks thru separate pointers */
  YYSTYPE *yyvs = yyvsa;	/*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YYLSP_NEEDED
  YYLTYPE yylsa[YYINITDEPTH];	/*  the location stack			*/
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;
  int yyfree_stacks = 0;

#ifdef YYPURE
  int yychar;
  YYSTYPE yylval;
  int yynerrs;
#ifdef YYLSP_NEEDED
  YYLTYPE yylloc;
#endif
#endif

  YYSTYPE yyval;		/*  the variable used to return		*/
				/*  semantic values from the action	*/
				/*  routines				*/

  int yylen;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Starting parse\n");
#endif

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YYLSP_NEEDED
  yylsp = yyls;
#endif

/* Push a new state, which is found in  yystate  .  */
/* In all cases, when you get here, the value and location stacks
   have just been pushed. so pushing a state here evens the stacks.  */
yynewstate:

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1)
    {
      /* Give user a chance to reallocate the stack */
      /* Use copies of these so that the &'s don't force the real ones into memory. */
      YYSTYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YYLSP_NEEDED
      YYLTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
	 the data in use in that stack, in bytes.  */
#ifdef YYLSP_NEEDED
      /* This used to be a conditional around just the two extra args,
	 but that might be undefined if yyoverflow is a macro.  */
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yyls1, size * sizeof (*yylsp),
		 &yystacksize);
#else
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yystacksize);
#endif

      yyss = yyss1; yyvs = yyvs1;
#ifdef YYLSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
	{
	  yyerror("parser stack overflow");
	  if (yyfree_stacks)
	    {
	      free (yyss);
	      free (yyvs);
#ifdef YYLSP_NEEDED
	      free (yyls);
#endif
	    }
	  return 2;
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
#ifndef YYSTACK_USE_ALLOCA
      yyfree_stacks = 1;
#endif
      yyss = (short *) YYSTACK_ALLOC (yystacksize * sizeof (*yyssp));
      __yy_memcpy ((char *)yyss, (char *)yyss1,
		   size * (unsigned int) sizeof (*yyssp));
      yyvs = (YYSTYPE *) YYSTACK_ALLOC (yystacksize * sizeof (*yyvsp));
      __yy_memcpy ((char *)yyvs, (char *)yyvs1,
		   size * (unsigned int) sizeof (*yyvsp));
#ifdef YYLSP_NEEDED
      yyls = (YYLTYPE *) YYSTACK_ALLOC (yystacksize * sizeof (*yylsp));
      __yy_memcpy ((char *)yyls, (char *)yyls1,
		   size * (unsigned int) sizeof (*yylsp));
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YYLSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
	YYABORT;
    }

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  goto yybackup;
 yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (yychar == YYEMPTY)
    {
#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Reading a token: ");
#endif
      yychar = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (yychar <= 0)		/* This means end of input. */
    {
      yychar1 = 0;
      yychar = YYEOF;		/* Don't call YYLEX any more */

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(yychar);

#if YYDEBUG != 0
      if (yydebug)
	{
	  fprintf (stderr, "Next token is %d (%s", yychar, yytname[yychar1]);
	  /* Give the individual parser a way to print the precise meaning
	     of a token, for further debugging info.  */
#ifdef YYPRINT
	  YYPRINT (stderr, yychar, yylval);
#endif
	  fprintf (stderr, ")\n");
	}
#endif
    }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    goto yydefault;

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrlab;

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting token %d (%s), ", yychar, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus) yyerrstatus--;

  yystate = yyn;
  goto yynewstate;

/* Do the default action for the current state.  */
yydefault:

  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;

/* Do a reduction.  yyn is the number of a rule to reduce with.  */
yyreduce:
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1-yylen]; /* implement default value of the action */

#if YYDEBUG != 0
  if (yydebug)
    {
      int i;

      fprintf (stderr, "Reducing via rule %d (line %d), ",
	       yyn, yyrline[yyn]);

      /* Print the symbols being reduced, and their result.  */
      for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
	fprintf (stderr, "%s ", yytname[yyrhs[i]]);
      fprintf (stderr, " -> %s\n", yytname[yyr1[yyn]]);
    }
#endif


  switch (yyn) {

case 1:
#line 223 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Rec = Records.getClass(*yyvsp[0].StrVal);
    if (yyval.Rec == 0) {
      err() << "Couldn't find class '" << *yyvsp[0].StrVal << "'!\n";
      exit(1);
    }
    delete yyvsp[0].StrVal;
  ;
    break;}
case 2:
#line 234 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{                       // string type
    yyval.Ty = new StringRecTy();
  ;
    break;}
case 3:
#line 236 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{                           // bit type
    yyval.Ty = new BitRecTy();
  ;
    break;}
case 4:
#line 238 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{           // bits<x> type
    yyval.Ty = new BitsRecTy(yyvsp[-1].IntVal);
  ;
    break;}
case 5:
#line 240 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{                           // int type
    yyval.Ty = new IntRecTy();
  ;
    break;}
case 6:
#line 242 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{          // list<x> type
    yyval.Ty = new ListRecTy(yyvsp[-1].Ty);
  ;
    break;}
case 7:
#line 244 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{                          // code type
    yyval.Ty = new CodeRecTy();
  ;
    break;}
case 8:
#line 246 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{                           // dag type
    yyval.Ty = new DagRecTy();
  ;
    break;}
case 9:
#line 248 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{                       // Record Type
    yyval.Ty = new RecordRecTy(yyvsp[0].Rec);
  ;
    break;}
case 10:
#line 252 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.IntVal = 0; ;
    break;}
case 11:
#line 252 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.IntVal = 1; ;
    break;}
case 12:
#line 254 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.Initializer = 0; ;
    break;}
case 13:
#line 254 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.Initializer = yyvsp[0].Initializer; ;
    break;}
case 14:
#line 256 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = new IntInit(yyvsp[0].IntVal);
  ;
    break;}
case 15:
#line 258 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = new StringInit(*yyvsp[0].StrVal);
    delete yyvsp[0].StrVal;
  ;
    break;}
case 16:
#line 261 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = new CodeInit(*yyvsp[0].StrVal);
    delete yyvsp[0].StrVal;
  ;
    break;}
case 17:
#line 264 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = new UnsetInit();
  ;
    break;}
case 18:
#line 266 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 19:
#line 279 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 20:
#line 307 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 21:
#line 322 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = yyvsp[-3].Initializer->convertInitializerBitRange(*yyvsp[-1].BitList);
    if (yyval.Initializer == 0) {
      err() << "Invalid bit range for value '" << *yyvsp[-3].Initializer << "'!\n";
      exit(1);
    }
    delete yyvsp[-1].BitList;
  ;
    break;}
case 22:
#line 329 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = new ListInit(*yyvsp[-1].FieldList);
    delete yyvsp[-1].FieldList;
  ;
    break;}
case 23:
#line 332 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    if (!yyvsp[-2].Initializer->getFieldType(*yyvsp[0].StrVal)) {
      err() << "Cannot access field '" << *yyvsp[0].StrVal << "' of value '" << *yyvsp[-2].Initializer << "!\n";
      exit(1);
    }
    yyval.Initializer = new FieldInit(yyvsp[-2].Initializer, *yyvsp[0].StrVal);
    delete yyvsp[0].StrVal;
  ;
    break;}
case 24:
#line 339 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    Record *D = Records.getDef(*yyvsp[-2].StrVal);
    if (D == 0) {
      err() << "Invalid def '" << *yyvsp[-2].StrVal << "'!\n";
      exit(1);
    }
    yyval.Initializer = new DagInit(D, *yyvsp[-1].DagValueList);
    delete yyvsp[-2].StrVal; delete yyvsp[-1].DagValueList;
  ;
    break;}
case 25:
#line 347 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    std::reverse(yyvsp[-1].BitList->begin(), yyvsp[-1].BitList->end());
    yyval.Initializer = yyvsp[-3].Initializer->convertInitListSlice(*yyvsp[-1].BitList);
    if (yyval.Initializer == 0) {
      err() << "Invalid list slice for value '" << *yyvsp[-3].Initializer << "'!\n";
      exit(1);
    }
    delete yyvsp[-1].BitList;
  ;
    break;}
case 26:
#line 355 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = yyvsp[-3].Initializer->getBinaryOp(Init::SHL, yyvsp[-1].Initializer);
    if (yyval.Initializer == 0) {
      err() << "Cannot shift values '" << *yyvsp[-3].Initializer << "' and '" << *yyvsp[-1].Initializer << "'!\n";
      exit(1);
    }
  ;
    break;}
case 27:
#line 361 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = yyvsp[-3].Initializer->getBinaryOp(Init::SRA, yyvsp[-1].Initializer);
    if (yyval.Initializer == 0) {
      err() << "Cannot shift values '" << *yyvsp[-3].Initializer << "' and '" << *yyvsp[-1].Initializer << "'!\n";
      exit(1);
    }
  ;
    break;}
case 28:
#line 367 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.Initializer = yyvsp[-3].Initializer->getBinaryOp(Init::SRL, yyvsp[-1].Initializer);
    if (yyval.Initializer == 0) {
      err() << "Cannot shift values '" << *yyvsp[-3].Initializer << "' and '" << *yyvsp[-1].Initializer << "'!\n";
      exit(1);
    }
  ;
    break;}
case 29:
#line 375 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.StrVal = new std::string();
  ;
    break;}
case 30:
#line 378 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.StrVal = yyvsp[0].StrVal;
  ;
    break;}
case 31:
#line 382 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.DagValueList = new std::vector<std::pair<Init*, std::string> >();
    yyval.DagValueList->push_back(std::make_pair(yyvsp[-1].Initializer, *yyvsp[0].StrVal));
    delete yyvsp[0].StrVal;
  ;
    break;}
case 32:
#line 387 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyvsp[-3].DagValueList->push_back(std::make_pair(yyvsp[-1].Initializer, *yyvsp[0].StrVal));
    delete yyvsp[0].StrVal;
    yyval.DagValueList = yyvsp[-3].DagValueList;
  ;
    break;}
case 33:
#line 393 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.DagValueList = new std::vector<std::pair<Init*, std::string> >();
  ;
    break;}
case 34:
#line 396 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.DagValueList = yyvsp[0].DagValueList; ;
    break;}
case 35:
#line 399 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.BitList = new std::vector<unsigned>();
    yyval.BitList->push_back(yyvsp[0].IntVal);
  ;
    break;}
case 36:
#line 402 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 37:
#line 415 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 38:
#line 429 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    (yyval.BitList=yyvsp[-2].BitList)->push_back(yyvsp[0].IntVal);
  ;
    break;}
case 39:
#line 431 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 40:
#line 444 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
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
  ;
    break;}
case 41:
#line 460 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.BitList = yyvsp[0].BitList; std::reverse(yyvsp[0].BitList->begin(), yyvsp[0].BitList->end()); ;
    break;}
case 42:
#line 462 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.BitList = 0; ;
    break;}
case 43:
#line 462 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.BitList = yyvsp[-1].BitList; ;
    break;}
case 44:
#line 466 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.FieldList = new std::vector<Init*>();
  ;
    break;}
case 45:
#line 468 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.FieldList = yyvsp[0].FieldList;
  ;
    break;}
case 46:
#line 472 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.FieldList = new std::vector<Init*>();
    yyval.FieldList->push_back(yyvsp[0].Initializer);
  ;
    break;}
case 47:
#line 475 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    (yyval.FieldList = yyvsp[-2].FieldList)->push_back(yyvsp[0].Initializer);
  ;
    break;}
case 48:
#line 479 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  std::string DecName = *yyvsp[-1].StrVal;
  if (ParsingTemplateArgs)
    DecName = CurRec->getName() + ":" + DecName;

  addValue(RecordVal(DecName, yyvsp[-2].Ty, yyvsp[-3].IntVal));
  setValue(DecName, 0, yyvsp[0].Initializer);
  yyval.StrVal = new std::string(DecName);
;
    break;}
case 49:
#line 489 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  delete yyvsp[-1].StrVal;
;
    break;}
case 50:
#line 491 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  setValue(*yyvsp[-4].StrVal, yyvsp[-3].BitList, yyvsp[-1].Initializer);
  delete yyvsp[-4].StrVal;
  delete yyvsp[-3].BitList;
;
    break;}
case 55:
#line 500 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.SubClassRef = new SubClassRefTy(yyvsp[0].Rec, new std::vector<Init*>());
  ;
    break;}
case 56:
#line 502 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.SubClassRef = new SubClassRefTy(yyvsp[-3].Rec, yyvsp[-1].FieldList);
  ;
    break;}
case 57:
#line 506 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.SubClassList = new std::vector<SubClassRefTy>();
    yyval.SubClassList->push_back(*yyvsp[0].SubClassRef);
    delete yyvsp[0].SubClassRef;
  ;
    break;}
case 58:
#line 511 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    (yyval.SubClassList=yyvsp[-2].SubClassList)->push_back(*yyvsp[0].SubClassRef);
    delete yyvsp[0].SubClassRef;
  ;
    break;}
case 59:
#line 516 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.SubClassList = new std::vector<SubClassRefTy>();
  ;
    break;}
case 60:
#line 519 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    yyval.SubClassList = yyvsp[0].SubClassList;
  ;
    break;}
case 61:
#line 523 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  CurRec->addTemplateArg(*yyvsp[0].StrVal);
  delete yyvsp[0].StrVal;
;
    break;}
case 62:
#line 526 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  CurRec->addTemplateArg(*yyvsp[0].StrVal);
  delete yyvsp[0].StrVal;
;
    break;}
case 63:
#line 531 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{;
    break;}
case 66:
#line 534 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.StrVal = yyvsp[0].StrVal; ;
    break;}
case 67:
#line 534 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ yyval.StrVal = new std::string(); ;
    break;}
case 68:
#line 536 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  static unsigned AnonCounter = 0;
  if (yyvsp[0].StrVal->empty())
    *yyvsp[0].StrVal = "anonymous."+utostr(AnonCounter++);
  yyval.StrVal = yyvsp[0].StrVal;
;
    break;}
case 69:
#line 543 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  // If a class of this name already exists, it must be a forward ref.
  if (CurRec = Records.getClass(*yyvsp[0].StrVal)) {
    // If the body was previously defined, this is an error.
    if (!CurRec->getValues().empty() ||
        !CurRec->getSuperClasses().empty() ||
        !CurRec->getTemplateArgs().empty()) {
      err() << "Class '" << CurRec->getName() << "' already defined!\n";
      exit(1);
    }
  } else {
    // If this is the first reference to this class, create and add it.
    CurRec = new Record(*yyvsp[0].StrVal);
    Records.addClass(CurRec);
  }
  delete yyvsp[0].StrVal;
;
    break;}
case 70:
#line 561 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  CurRec = new Record(*yyvsp[0].StrVal);
  delete yyvsp[0].StrVal;
  
  // Ensure redefinition doesn't happen.
  if (Records.getDef(CurRec->getName())) {
    err() << "Def '" << CurRec->getName() << "' already defined!\n";
    exit(1);
  }
  Records.addDef(CurRec);
;
    break;}
case 71:
#line 573 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
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
         ;
    break;}
case 72:
#line 587 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
           yyval.Rec = CurRec;
           CurRec = 0;
         ;
    break;}
case 73:
#line 592 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
                ParsingTemplateArgs = true;
            ;
    break;}
case 74:
#line 594 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
                ParsingTemplateArgs = false;
            ;
    break;}
case 75:
#line 596 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
        yyval.Rec = yyvsp[0].Rec;
     ;
    break;}
case 76:
#line 600 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  yyvsp[0].Rec->resolveReferences();

  // If ObjectBody has template arguments, it's an error.
  assert(yyvsp[0].Rec->getTemplateArgs().empty() && "How'd this get template args?");
  yyval.Rec = yyvsp[0].Rec;
;
    break;}
case 79:
#line 611 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
  LetStack.back().push_back(LetRecord(*yyvsp[-3].StrVal, yyvsp[-2].BitList, yyvsp[0].Initializer));
  delete yyvsp[-3].StrVal; delete yyvsp[-2].BitList;
;
    break;}
case 82:
#line 619 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{ LetStack.push_back(std::vector<LetRecord>()); ;
    break;}
case 84:
#line 622 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    LetStack.pop_back();
  ;
    break;}
case 85:
#line 625 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{
    LetStack.pop_back();
  ;
    break;}
case 86:
#line 629 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{;
    break;}
case 87:
#line 629 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{;
    break;}
case 88:
#line 631 "/Users/sabre/llvm/utils/TableGen/FileParser.y"
{;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 543 "/usr/share/bison.simple"

  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YYLSP_NEEDED
  yylsp -= yylen;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YYLSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = yylloc.first_line;
      yylsp->first_column = yylloc.first_column;
      yylsp->last_line = (yylsp-1)->last_line;
      yylsp->last_column = (yylsp-1)->last_column;
      yylsp->text = 0;
    }
  else
    {
      yylsp->last_line = (yylsp+yylen-1)->last_line;
      yylsp->last_column = (yylsp+yylen-1)->last_column;
    }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  goto yynewstate;

yyerrlab:   /* here on detecting error */

  if (! yyerrstatus)
    /* If not already recovering from an error, report this error.  */
    {
      ++yynerrs;

#ifdef YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (yyn > YYFLAG && yyn < YYLAST)
	{
	  int size = 0;
	  char *msg;
	  int x, count;

	  count = 0;
	  /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
	  for (x = (yyn < 0 ? -yyn : 0);
	       x < (sizeof(yytname) / sizeof(char *)); x++)
	    if (yycheck[x + yyn] == x)
	      size += strlen(yytname[x]) + 15, count++;
	  msg = (char *) malloc(size + 15);
	  if (msg != 0)
	    {
	      strcpy(msg, "parse error");

	      if (count < 5)
		{
		  count = 0;
		  for (x = (yyn < 0 ? -yyn : 0);
		       x < (sizeof(yytname) / sizeof(char *)); x++)
		    if (yycheck[x + yyn] == x)
		      {
			strcat(msg, count == 0 ? ", expecting `" : " or `");
			strcat(msg, yytname[x]);
			strcat(msg, "'");
			count++;
		      }
		}
	      yyerror(msg);
	      free(msg);
	    }
	  else
	    yyerror ("parse error; also virtual memory exceeded");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror("parse error");
    }

  goto yyerrlab1;
yyerrlab1:   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (yychar == YYEOF)
	YYABORT;

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Discarding token %d (%s).\n", yychar, yytname[yychar1]);
#endif

      yychar = YYEMPTY;
    }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3;		/* Each real token shifted decrements this */

  goto yyerrhandle;

yyerrdefault:  /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) goto yydefault;
#endif

yyerrpop:   /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss) YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YYLSP_NEEDED
  yylsp--;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "Error: state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

yyerrhandle:

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yyerrdefault;

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    goto yyerrdefault;

  yyn = yytable[yyn];
  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrpop;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrpop;

  if (yyn == YYFINAL)
    YYACCEPT;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  yystate = yyn;
  goto yynewstate;

 yyacceptlab:
  /* YYACCEPT comes here.  */
  if (yyfree_stacks)
    {
      free (yyss);
      free (yyvs);
#ifdef YYLSP_NEEDED
      free (yyls);
#endif
    }
  return 0;

 yyabortlab:
  /* YYABORT comes here.  */
  if (yyfree_stacks)
    {
      free (yyss);
      free (yyvs);
#ifdef YYLSP_NEEDED
      free (yyls);
#endif
    }
  return 1;
}
#line 633 "/Users/sabre/llvm/utils/TableGen/FileParser.y"


int yyerror(const char *ErrorMsg) {
  err() << "Error parsing: " << ErrorMsg << "\n";
  exit(1);
}
