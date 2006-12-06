
/*  A Bison parser, made from /Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define yyparse Upgradeparse
#define yylex Upgradelex
#define yyerror Upgradeerror
#define yylval Upgradelval
#define yychar Upgradechar
#define yydebug Upgradedebug
#define yynerrs Upgradenerrs
#define	VOID	257
#define	BOOL	258
#define	SBYTE	259
#define	UBYTE	260
#define	SHORT	261
#define	USHORT	262
#define	INT	263
#define	UINT	264
#define	LONG	265
#define	ULONG	266
#define	FLOAT	267
#define	DOUBLE	268
#define	LABEL	269
#define	OPAQUE	270
#define	ESINT64VAL	271
#define	EUINT64VAL	272
#define	SINTVAL	273
#define	UINTVAL	274
#define	FPVAL	275
#define	NULL_TOK	276
#define	UNDEF	277
#define	ZEROINITIALIZER	278
#define	TRUETOK	279
#define	FALSETOK	280
#define	TYPE	281
#define	VAR_ID	282
#define	LABELSTR	283
#define	STRINGCONSTANT	284
#define	IMPLEMENTATION	285
#define	BEGINTOK	286
#define	ENDTOK	287
#define	DECLARE	288
#define	GLOBAL	289
#define	CONSTANT	290
#define	SECTION	291
#define	VOLATILE	292
#define	TO	293
#define	DOTDOTDOT	294
#define	CONST	295
#define	INTERNAL	296
#define	LINKONCE	297
#define	WEAK	298
#define	DLLIMPORT	299
#define	DLLEXPORT	300
#define	EXTERN_WEAK	301
#define	APPENDING	302
#define	NOT	303
#define	EXTERNAL	304
#define	TARGET	305
#define	TRIPLE	306
#define	ENDIAN	307
#define	POINTERSIZE	308
#define	LITTLE	309
#define	BIG	310
#define	ALIGN	311
#define	UNINITIALIZED	312
#define	DEPLIBS	313
#define	CALL	314
#define	TAIL	315
#define	ASM_TOK	316
#define	MODULE	317
#define	SIDEEFFECT	318
#define	CC_TOK	319
#define	CCC_TOK	320
#define	CSRETCC_TOK	321
#define	FASTCC_TOK	322
#define	COLDCC_TOK	323
#define	X86_STDCALLCC_TOK	324
#define	X86_FASTCALLCC_TOK	325
#define	DATALAYOUT	326
#define	RET	327
#define	BR	328
#define	SWITCH	329
#define	INVOKE	330
#define	EXCEPT	331
#define	UNWIND	332
#define	UNREACHABLE	333
#define	ADD	334
#define	SUB	335
#define	MUL	336
#define	DIV	337
#define	UDIV	338
#define	SDIV	339
#define	FDIV	340
#define	REM	341
#define	UREM	342
#define	SREM	343
#define	FREM	344
#define	AND	345
#define	OR	346
#define	XOR	347
#define	SETLE	348
#define	SETGE	349
#define	SETLT	350
#define	SETGT	351
#define	SETEQ	352
#define	SETNE	353
#define	ICMP	354
#define	FCMP	355
#define	EQ	356
#define	NE	357
#define	SLT	358
#define	SGT	359
#define	SLE	360
#define	SGE	361
#define	OEQ	362
#define	ONE	363
#define	OLT	364
#define	OGT	365
#define	OLE	366
#define	OGE	367
#define	ORD	368
#define	UNO	369
#define	UEQ	370
#define	UNE	371
#define	ULT	372
#define	UGT	373
#define	ULE	374
#define	UGE	375
#define	MALLOC	376
#define	ALLOCA	377
#define	FREE	378
#define	LOAD	379
#define	STORE	380
#define	GETELEMENTPTR	381
#define	PHI_TOK	382
#define	SELECT	383
#define	SHL	384
#define	SHR	385
#define	ASHR	386
#define	LSHR	387
#define	VAARG	388
#define	EXTRACTELEMENT	389
#define	INSERTELEMENT	390
#define	SHUFFLEVECTOR	391
#define	CAST	392
#define	TRUNC	393
#define	ZEXT	394
#define	SEXT	395
#define	FPTRUNC	396
#define	FPEXT	397
#define	FPTOUI	398
#define	FPTOSI	399
#define	UITOFP	400
#define	SITOFP	401
#define	PTRTOINT	402
#define	INTTOPTR	403
#define	BITCAST	404

#line 14 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"

#include "ParserInternals.h"
#include <llvm/ADT/StringExtras.h>
#include <algorithm>
#include <map>
#include <utility>
#include <iostream>
#include <cassert>

#define YYERROR_VERBOSE 1
#define YYINCLUDED_STDLIB_H
#define YYDEBUG 1
#define UPGRADE_SETCOND_OPS 0

int yylex();                       // declaration" of xxx warnings.
int yyparse();
extern int yydebug;

static std::string CurFilename;
static std::ostream *O = 0;
std::istream* LexInput = 0;
unsigned SizeOfPointer = 32;
static uint64_t unique = 1;

typedef std::vector<TypeInfo> TypeVector;
static TypeVector EnumeratedTypes;
typedef std::map<std::string,TypeInfo> TypeMap;
static TypeMap NamedTypes;
static TypeMap Globals;

void destroy(ValueList* VL) {
  while (!VL->empty()) {
    ValueInfo& VI = VL->back();
    VI.destroy();
    VL->pop_back();
  }
  delete VL;
}

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

static void ResolveType(TypeInfo& Ty) {
  if (Ty.oldTy == UnresolvedTy) {
    TypeMap::iterator I = NamedTypes.find(*Ty.newTy);
    if (I != NamedTypes.end()) {
      Ty.oldTy = I->second.oldTy;
      Ty.elemTy = I->second.elemTy;
    } else {
      std::string msg("Can't resolve type: ");
      msg += *Ty.newTy;
      yyerror(msg.c_str());
    }
  } else if (Ty.oldTy == NumericTy) {
    unsigned ref = atoi(&((Ty.newTy->c_str())[1])); // Skip the '\\'
    if (ref < EnumeratedTypes.size()) {
      Ty.oldTy = EnumeratedTypes[ref].oldTy;
      Ty.elemTy = EnumeratedTypes[ref].elemTy;
    } else {
      std::string msg("Can't resolve type: ");
      msg += *Ty.newTy;
      yyerror(msg.c_str());
    }
  }
  // otherwise its already resolved.
}

static const char* getCastOpcode(
  std::string& Source, const TypeInfo& SrcTy, const TypeInfo& DstTy) 
{
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
      assert(!"Casting invalid type to pointer");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }
  return opcode;
}

static std::string getCastUpgrade(
  const std::string& Src, TypeInfo& SrcTy, TypeInfo& DstTy, bool isConst)
{
  std::string Result;
  std::string Source = Src;
  if (SrcTy.isFloatingPoint() && DstTy.isPointer()) {
    // fp -> ptr cast is no longer supported but we must upgrade this
    // by doing a double cast: fp -> int -> ptr
    if (isConst)
      Source = "ulong fptoui(" + Source + " to ulong)";
    else {
      *O << "    %cast_upgrade" << unique << " = fptoui " << Source 
         << " to ulong\n";
      Source = "ulong %cast_upgrade" + llvm::utostr(unique);
    }
    // Update the SrcTy for the getCastOpcode call below
    SrcTy.destroy();
    SrcTy.newTy = new std::string("ulong");
    SrcTy.oldTy = ULongTy;
  } else if (DstTy.oldTy == BoolTy) {
    // cast ptr %x to  bool was previously defined as setne ptr %x, null
    // The ptrtoint semantic is to truncate, not compare so we must retain
    // the original intent by replace the cast with a setne
    const char* comparator = SrcTy.isPointer() ? ", null" : 
      (SrcTy.isFloatingPoint() ? ", 0.0" : ", 0");
    if (isConst) 
      Result = "setne (" + Source + comparator + ")";
    else
      Result = "setne " + Source + comparator;
    return Result; // skip cast processing below
  }
  ResolveType(SrcTy);
  ResolveType(DstTy);
  std::string Opcode(getCastOpcode(Source, SrcTy, DstTy));
  if (isConst)
    Result += Opcode + "( " + Source + " to " + *DstTy.newTy + ")";
  else
    Result += Opcode + " " + Source + " to " + *DstTy.newTy;
  return Result;
}

const char* getDivRemOpcode(const std::string& opcode, const TypeInfo& TI) {
  const char* op = opcode.c_str();
  TypeInfo Ty = TI;
  ResolveType(Ty);
  if (Ty.isPacked())
    Ty.oldTy = Ty.getElementType();
  if (opcode == "div")
    if (Ty.isFloatingPoint())
      op = "fdiv";
    else if (Ty.isUnsigned())
      op = "udiv";
    else if (Ty.isSigned())
      op = "sdiv";
    else
      yyerror("Invalid type for div instruction");
  else if (opcode == "rem")
    if (Ty.isFloatingPoint())
      op = "frem";
    else if (Ty.isUnsigned())
      op = "urem";
    else if (Ty.isSigned())
      op = "srem";
    else
      yyerror("Invalid type for rem instruction");
  return op;
}

std::string 
getCompareOp(const std::string& setcc, const TypeInfo& TI) {
  assert(setcc.length() == 5);
  char cc1 = setcc[3];
  char cc2 = setcc[4];
  assert(cc1 == 'e' || cc1 == 'n' || cc1 == 'l' || cc1 == 'g');
  assert(cc2 == 'q' || cc2 == 'e' || cc2 == 'e' || cc2 == 't');
  std::string result("xcmp xxx");
  result[6] = cc1;
  result[7] = cc2;
  if (TI.isFloatingPoint()) {
    result[0] = 'f';
    result[5] = 'o'; // FIXME: Always map to ordered comparison ?
  } else if (TI.isIntegral() || TI.isPointer()) {
    result[0] = 'i';
    if ((cc1 == 'e' && cc2 == 'q') || (cc1 == 'n' && cc2 == 'e'))
      result.erase(5,1);
    else if (TI.isSigned())
      result[5] = 's';
    else if (TI.isUnsigned() || TI.isPointer() || TI.isBool())
      result[5] = 'u';
    else
      yyerror("Invalid integral type for setcc");
  }
  return result;
}


#line 270 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
typedef union {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		582
#define	YYFLAG		-32768
#define	YYNTBASE	165

#define YYTRANSLATE(x) ((unsigned)(x) <= 404 ? yytranslate[x] : 239)

static const short yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,   154,
   155,   163,     2,   152,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,   159,
   151,   160,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
   156,   153,   158,     2,     2,     2,     2,     2,   164,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,   157,
     2,     2,   161,     2,   162,     2,     2,     2,     2,     2,
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
    17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
    27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
    37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
    57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
    77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
    87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
    97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
   107,   108,   109,   110,   111,   112,   113,   114,   115,   116,
   117,   118,   119,   120,   121,   122,   123,   124,   125,   126,
   127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
   137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
   147,   148,   149,   150
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     4,     6,     8,    10,    12,    14,    16,    18,
    20,    22,    24,    26,    28,    30,    32,    34,    36,    38,
    40,    42,    44,    46,    48,    50,    52,    54,    56,    58,
    60,    62,    64,    66,    68,    70,    72,    74,    76,    78,
    80,    82,    84,    86,    88,    90,    92,    94,    96,    98,
   100,   102,   104,   106,   108,   110,   112,   114,   116,   118,
   120,   122,   124,   126,   128,   130,   132,   134,   136,   138,
   140,   142,   144,   146,   148,   150,   152,   154,   156,   158,
   161,   162,   164,   166,   168,   170,   172,   174,   176,   177,
   179,   181,   183,   185,   187,   189,   192,   193,   194,   197,
   198,   202,   205,   206,   208,   209,   213,   215,   218,   220,
   222,   224,   226,   228,   230,   232,   234,   236,   238,   240,
   242,   244,   246,   248,   250,   252,   254,   256,   258,   261,
   266,   272,   278,   282,   285,   288,   290,   294,   296,   300,
   302,   303,   308,   312,   316,   321,   326,   330,   333,   336,
   339,   342,   345,   348,   351,   354,   357,   360,   367,   373,
   382,   389,   396,   403,   411,   419,   426,   433,   442,   451,
   455,   457,   459,   461,   463,   466,   469,   474,   477,   479,
   481,   483,   488,   491,   496,   503,   510,   517,   524,   528,
   533,   534,   536,   538,   540,   544,   548,   552,   556,   560,
   564,   566,   567,   569,   571,   573,   574,   577,   581,   583,
   585,   589,   591,   592,   601,   603,   605,   609,   611,   613,
   617,   618,   620,   622,   626,   627,   629,   631,   633,   635,
   637,   639,   641,   643,   645,   649,   651,   657,   659,   661,
   663,   665,   668,   671,   673,   676,   679,   680,   682,   684,
   686,   689,   692,   696,   706,   716,   725,   740,   742,   744,
   751,   757,   760,   767,   775,   777,   781,   783,   784,   787,
   789,   795,   801,   807,   815,   823,   826,   831,   836,   843,
   848,   853,   860,   867,   870,   878,   880,   883,   884,   886,
   887,   891,   898,   902,   909,   912,   917,   924
};

static const short yyrhs[] = {    19,
     0,    20,     0,    17,     0,    18,     0,    80,     0,    81,
     0,    82,     0,    83,     0,    84,     0,    85,     0,    86,
     0,    87,     0,    88,     0,    89,     0,    90,     0,    91,
     0,    92,     0,    93,     0,    94,     0,    95,     0,    96,
     0,    97,     0,    98,     0,    99,     0,   102,     0,   103,
     0,   104,     0,   105,     0,   106,     0,   107,     0,   118,
     0,   119,     0,   120,     0,   121,     0,   108,     0,   109,
     0,   110,     0,   111,     0,   112,     0,   113,     0,   114,
     0,   115,     0,   116,     0,   117,     0,   118,     0,   119,
     0,   120,     0,   121,     0,    25,     0,    26,     0,   130,
     0,   131,     0,   132,     0,   133,     0,   139,     0,   140,
     0,   141,     0,   142,     0,   143,     0,   144,     0,   145,
     0,   146,     0,   147,     0,   148,     0,   149,     0,   150,
     0,   138,     0,    11,     0,     9,     0,     7,     0,     5,
     0,    12,     0,    10,     0,     8,     0,     6,     0,   174,
     0,   175,     0,    13,     0,    14,     0,   207,   151,     0,
     0,    42,     0,    43,     0,    44,     0,    48,     0,    45,
     0,    46,     0,    47,     0,     0,    66,     0,    67,     0,
    68,     0,    69,     0,    70,     0,    71,     0,    65,    18,
     0,     0,     0,    57,    18,     0,     0,   152,    57,    18,
     0,    37,    30,     0,     0,   183,     0,     0,   152,   186,
   185,     0,   183,     0,    57,    18,     0,   189,     0,     3,
     0,   191,     0,     3,     0,   191,     0,     4,     0,     5,
     0,     6,     0,     7,     0,     8,     0,     9,     0,    10,
     0,    11,     0,    12,     0,    13,     0,    14,     0,    15,
     0,    16,     0,   221,     0,   190,     0,   153,    18,     0,
   188,   154,   193,   155,     0,   156,    18,   157,   191,   158,
     0,   159,    18,   157,   191,   160,     0,   161,   192,   162,
     0,   161,   162,     0,   191,   163,     0,   191,     0,   192,
   152,   191,     0,   192,     0,   192,   152,    40,     0,    40,
     0,     0,   189,   156,   196,   158,     0,   189,   156,   158,
     0,   189,   164,    30,     0,   189,   159,   196,   160,     0,
   189,   161,   196,   162,     0,   189,   161,   162,     0,   189,
    22,     0,   189,    23,     0,   189,   221,     0,   189,   195,
     0,   189,    24,     0,   174,   166,     0,   175,    18,     0,
     4,    25,     0,     4,    26,     0,   177,    21,     0,   173,
   154,   194,    39,   189,   155,     0,   127,   154,   194,   236,
   155,     0,   129,   154,   194,   152,   194,   152,   194,   155,
     0,   167,   154,   194,   152,   194,   155,     0,   168,   154,
   194,   152,   194,   155,     0,   169,   154,   194,   152,   194,
   155,     0,   100,   170,   154,   194,   152,   194,   155,     0,
   101,   171,   154,   194,   152,   194,   155,     0,   172,   154,
   194,   152,   194,   155,     0,   135,   154,   194,   152,   194,
   155,     0,   136,   154,   194,   152,   194,   152,   194,   155,
     0,   137,   154,   194,   152,   194,   152,   194,   155,     0,
   196,   152,   194,     0,   194,     0,    35,     0,    36,     0,
   199,     0,   199,   216,     0,   199,   218,     0,   199,    63,
    62,   202,     0,   199,    31,     0,   201,     0,    50,     0,
    58,     0,   201,   178,    27,   187,     0,   201,   218,     0,
   201,    63,    62,   202,     0,   201,   178,   179,   197,   194,
   185,     0,   201,   178,   200,   197,   189,   185,     0,   201,
   178,    45,   197,   189,   185,     0,   201,   178,    47,   197,
   189,   185,     0,   201,    51,   204,     0,   201,    59,   151,
   205,     0,     0,    30,     0,    56,     0,    55,     0,    53,
   151,   203,     0,    54,   151,    18,     0,    52,   151,    30,
     0,    72,   151,    30,     0,   156,   206,   158,     0,   206,
   152,    30,     0,    30,     0,     0,    28,     0,    30,     0,
   207,     0,     0,   189,   208,     0,   210,   152,   209,     0,
   209,     0,   210,     0,   210,   152,    40,     0,    40,     0,
     0,   180,   187,   207,   154,   211,   155,   184,   181,     0,
    32,     0,   161,     0,   179,   212,   213,     0,    33,     0,
   162,     0,   214,   224,   215,     0,     0,    45,     0,    47,
     0,    34,   217,   212,     0,     0,    64,     0,    17,     0,
    18,     0,    21,     0,    25,     0,    26,     0,    22,     0,
    23,     0,    24,     0,   159,   196,   160,     0,   195,     0,
    62,   219,    30,   152,    30,     0,   165,     0,   207,     0,
   221,     0,   220,     0,   189,   222,     0,   224,   225,     0,
   225,     0,   226,   228,     0,   226,   230,     0,     0,    29,
     0,    78,     0,    77,     0,    73,   223,     0,    73,     3,
     0,    74,    15,   222,     0,    74,     4,   222,   152,    15,
   222,   152,    15,   222,     0,    75,   176,   222,   152,    15,
   222,   156,   229,   158,     0,    75,   176,   222,   152,    15,
   222,   156,   158,     0,   178,    76,   180,   187,   222,   154,
   233,   155,    39,    15,   222,   227,    15,   222,     0,   227,
     0,    79,     0,   229,   176,   220,   152,    15,   222,     0,
   176,   220,   152,    15,   222,     0,   178,   235,     0,   189,
   156,   222,   152,   222,   158,     0,   231,   152,   156,   222,
   152,   222,   158,     0,   223,     0,   232,   152,   223,     0,
   232,     0,     0,    61,    60,     0,    60,     0,   167,   189,
   222,   152,   222,     0,   168,   189,   222,   152,   222,     0,
   169,   189,   222,   152,   222,     0,   100,   170,   189,   222,
   152,   222,   155,     0,   101,   171,   189,   222,   152,   222,
   155,     0,    49,   223,     0,   172,   223,   152,   223,     0,
   173,   223,    39,   189,     0,   129,   223,   152,   223,   152,
   223,     0,   134,   223,   152,   189,     0,   135,   223,   152,
   223,     0,   136,   223,   152,   223,   152,   223,     0,   137,
   223,   152,   223,   152,   223,     0,   128,   231,     0,   234,
   180,   187,   222,   154,   233,   155,     0,   238,     0,   152,
   232,     0,     0,    38,     0,     0,   122,   189,   182,     0,
   122,   189,   152,    10,   222,   182,     0,   123,   189,   182,
     0,   123,   189,   152,    10,   222,   182,     0,   124,   223,
     0,   237,   125,   189,   222,     0,   237,   126,   223,   152,
   189,   222,     0,   127,   189,   222,   236,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   333,   333,   334,   334,   338,   338,   338,   338,   338,   338,
   338,   339,   339,   339,   339,   340,   340,   340,   341,   341,
   341,   341,   341,   341,   342,   342,   342,   342,   342,   342,
   342,   342,   342,   342,   343,   343,   343,   343,   343,   343,
   343,   343,   343,   343,   344,   344,   344,   344,   344,   344,
   345,   345,   345,   345,   346,   346,   346,   346,   346,   346,
   346,   346,   347,   347,   347,   347,   347,   352,   352,   352,
   352,   353,   353,   353,   353,   354,   354,   355,   355,   358,
   361,   366,   366,   366,   366,   366,   366,   367,   368,   371,
   371,   371,   371,   371,   372,   373,   378,   383,   384,   387,
   388,   396,   402,   403,   406,   407,   416,   417,   430,   430,
   431,   431,   432,   436,   436,   436,   436,   436,   436,   436,
   437,   437,   437,   437,   437,   439,   443,   447,   450,   455,
   461,   469,   477,   483,   487,   498,   501,   509,   510,   515,
   518,   528,   534,   539,   545,   551,   557,   562,   568,   574,
   580,   586,   592,   598,   604,   610,   616,   624,   638,   650,
   655,   661,   666,   674,   679,   684,   692,   697,   702,   712,
   717,   722,   722,   732,   737,   740,   745,   749,   753,   755,
   755,   758,   770,   775,   780,   789,   798,   807,   816,   821,
   826,   831,   833,   833,   836,   841,   848,   853,   860,   867,
   872,   873,   881,   881,   882,   882,   884,   891,   895,   899,
   902,   907,   910,   912,   932,   933,   935,   944,   945,   947,
   955,   956,   957,   961,   974,   975,   978,   978,   978,   978,
   978,   978,   978,   979,   980,   985,   986,   995,   995,   999,
  1005,  1016,  1022,  1025,  1033,  1037,  1042,  1045,  1051,  1051,
  1053,  1058,  1063,  1068,  1076,  1083,  1089,  1109,  1114,  1120,
  1125,  1133,  1142,  1149,  1157,  1161,  1168,  1169,  1173,  1178,
  1181,  1187,  1192,  1200,  1205,  1210,  1215,  1223,  1237,  1242,
  1247,  1252,  1257,  1262,  1267,  1284,  1289,  1290,  1294,  1295,
  1298,  1305,  1312,  1319,  1326,  1331,  1338,  1345
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","VOID","BOOL",
"SBYTE","UBYTE","SHORT","USHORT","INT","UINT","LONG","ULONG","FLOAT","DOUBLE",
"LABEL","OPAQUE","ESINT64VAL","EUINT64VAL","SINTVAL","UINTVAL","FPVAL","NULL_TOK",
"UNDEF","ZEROINITIALIZER","TRUETOK","FALSETOK","TYPE","VAR_ID","LABELSTR","STRINGCONSTANT",
"IMPLEMENTATION","BEGINTOK","ENDTOK","DECLARE","GLOBAL","CONSTANT","SECTION",
"VOLATILE","TO","DOTDOTDOT","CONST","INTERNAL","LINKONCE","WEAK","DLLIMPORT",
"DLLEXPORT","EXTERN_WEAK","APPENDING","NOT","EXTERNAL","TARGET","TRIPLE","ENDIAN",
"POINTERSIZE","LITTLE","BIG","ALIGN","UNINITIALIZED","DEPLIBS","CALL","TAIL",
"ASM_TOK","MODULE","SIDEEFFECT","CC_TOK","CCC_TOK","CSRETCC_TOK","FASTCC_TOK",
"COLDCC_TOK","X86_STDCALLCC_TOK","X86_FASTCALLCC_TOK","DATALAYOUT","RET","BR",
"SWITCH","INVOKE","EXCEPT","UNWIND","UNREACHABLE","ADD","SUB","MUL","DIV","UDIV",
"SDIV","FDIV","REM","UREM","SREM","FREM","AND","OR","XOR","SETLE","SETGE","SETLT",
"SETGT","SETEQ","SETNE","ICMP","FCMP","EQ","NE","SLT","SGT","SLE","SGE","OEQ",
"ONE","OLT","OGT","OLE","OGE","ORD","UNO","UEQ","UNE","ULT","UGT","ULE","UGE",
"MALLOC","ALLOCA","FREE","LOAD","STORE","GETELEMENTPTR","PHI_TOK","SELECT","SHL",
"SHR","ASHR","LSHR","VAARG","EXTRACTELEMENT","INSERTELEMENT","SHUFFLEVECTOR",
"CAST","TRUNC","ZEXT","SEXT","FPTRUNC","FPEXT","FPTOUI","FPTOSI","UITOFP","SITOFP",
"PTRTOINT","INTTOPTR","BITCAST","'='","','","'\\\\'","'('","')'","'['","'x'",
"']'","'<'","'>'","'{'","'}'","'*'","'c'","IntVal","EInt64Val","ArithmeticOps",
"LogicalOps","SetCondOps","IPredicates","FPredicates","ShiftOps","CastOps","SIntType",
"UIntType","IntType","FPType","OptAssign","OptLinkage","OptCallingConv","OptAlign",
"OptCAlign","SectionString","OptSection","GlobalVarAttributes","GlobalVarAttribute",
"TypesV","UpRTypesV","Types","PrimType","UpRTypes","TypeListI","ArgTypeListI",
"ConstVal","ConstExpr","ConstVector","GlobalType","Module","DefinitionList",
"External","ConstPool","AsmBlock","BigOrLittle","TargetDefinition","LibrariesDefinition",
"LibList","Name","OptName","ArgVal","ArgListH","ArgList","FunctionHeaderH","BEGIN",
"FunctionHeader","END","Function","FnDeclareLinkage","FunctionProto","OptSideEffect",
"ConstValueRef","SymbolicValueRef","ValueRef","ResolvedVal","BasicBlockList",
"BasicBlock","InstructionList","Unwind","BBTerminatorInst","JumpTable","Inst",
"PHIList","ValueRefList","ValueRefListE","OptTailCall","InstVal","IndexList",
"OptVolatile","MemoryInst", NULL
};
#endif

static const short yyr1[] = {     0,
   165,   165,   166,   166,   167,   167,   167,   167,   167,   167,
   167,   167,   167,   167,   167,   168,   168,   168,   169,   169,
   169,   169,   169,   169,   170,   170,   170,   170,   170,   170,
   170,   170,   170,   170,   171,   171,   171,   171,   171,   171,
   171,   171,   171,   171,   171,   171,   171,   171,   171,   171,
   172,   172,   172,   172,   173,   173,   173,   173,   173,   173,
   173,   173,   173,   173,   173,   173,   173,   174,   174,   174,
   174,   175,   175,   175,   175,   176,   176,   177,   177,   178,
   178,   179,   179,   179,   179,   179,   179,   179,   179,   180,
   180,   180,   180,   180,   180,   180,   180,   181,   181,   182,
   182,   183,   184,   184,   185,   185,   186,   186,   187,   187,
   188,   188,   189,   190,   190,   190,   190,   190,   190,   190,
   190,   190,   190,   190,   190,   191,   191,   191,   191,   191,
   191,   191,   191,   191,   191,   192,   192,   193,   193,   193,
   193,   194,   194,   194,   194,   194,   194,   194,   194,   194,
   194,   194,   194,   194,   194,   194,   194,   195,   195,   195,
   195,   195,   195,   195,   195,   195,   195,   195,   195,   196,
   196,   197,   197,   198,   199,   199,   199,   199,   199,   200,
   200,   201,   201,   201,   201,   201,   201,   201,   201,   201,
   201,   202,   203,   203,   204,   204,   204,   204,   205,   206,
   206,   206,   207,   207,   208,   208,   209,   210,   210,   211,
   211,   211,   211,   212,   213,   213,   214,   215,   215,   216,
   217,   217,   217,   218,   219,   219,   220,   220,   220,   220,
   220,   220,   220,   220,   220,   220,   220,   221,   221,   222,
   222,   223,   224,   224,   225,   226,   226,   226,   227,   227,
   228,   228,   228,   228,   228,   228,   228,   228,   228,   229,
   229,   230,   231,   231,   232,   232,   233,   233,   234,   234,
   235,   235,   235,   235,   235,   235,   235,   235,   235,   235,
   235,   235,   235,   235,   235,   235,   236,   236,   237,   237,
   238,   238,   238,   238,   238,   238,   238,   238
};

static const short yyr2[] = {     0,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
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
     6,     6,     6,     7,     7,     6,     6,     8,     8,     3,
     1,     1,     1,     1,     2,     2,     4,     2,     1,     1,
     1,     4,     2,     4,     6,     6,     6,     6,     3,     4,
     0,     1,     1,     1,     3,     3,     3,     3,     3,     3,
     1,     0,     1,     1,     1,     0,     2,     3,     1,     1,
     3,     1,     0,     8,     1,     1,     3,     1,     1,     3,
     0,     1,     1,     3,     0,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     3,     1,     5,     1,     1,     1,
     1,     2,     2,     1,     2,     2,     0,     1,     1,     1,
     2,     2,     3,     9,     9,     8,    14,     1,     1,     6,
     5,     2,     6,     7,     1,     3,     1,     0,     2,     1,
     5,     5,     5,     7,     7,     2,     4,     4,     6,     4,
     4,     6,     6,     2,     7,     1,     2,     0,     1,     0,
     3,     6,     3,     6,     2,     4,     6,     4
};

static const short yydefact[] = {   191,
    89,   179,   178,   221,    82,    83,    84,    86,    87,    88,
    85,     0,    97,   247,   175,   176,   203,   204,     0,     0,
     0,    89,     0,   183,   222,   223,    97,     0,     0,    90,
    91,    92,    93,    94,    95,     0,     0,   248,   247,   244,
    81,     0,     0,     0,     0,   189,     0,     0,     0,     0,
     0,   180,   181,     0,     0,    80,   224,   192,   177,    96,
   110,   114,   115,   116,   117,   118,   119,   120,   121,   122,
   123,   124,   125,   126,     1,     2,     0,     0,     0,     0,
   238,     0,     0,   109,   128,   113,   239,   127,   215,   216,
   217,   218,   219,   220,   243,     0,     0,     0,   250,   249,
   259,   290,   258,   245,   246,     0,     0,     0,     0,   202,
   190,   184,   182,   172,   173,     0,     0,     0,     0,   129,
     0,     0,   112,   134,   136,     0,     0,   141,   135,   252,
     0,   251,     0,     0,    71,    75,    70,    74,    69,    73,
    68,    72,    76,    77,     0,   289,     0,   270,     0,    97,
     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
    15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
     0,     0,     0,     0,     0,     0,     0,     0,    51,    52,
    53,    54,     0,     0,     0,     0,    67,    55,    56,    57,
    58,    59,    60,    61,    62,    63,    64,    65,    66,     0,
     0,     0,     0,     0,    97,   262,     0,   286,   197,   194,
   193,   195,   196,   198,   201,     0,   105,   105,   114,   115,
   116,   117,   118,   119,   120,   121,   122,   123,   124,     0,
     0,     0,     0,   105,   105,     0,     0,     0,   133,   213,
   140,   138,     0,   227,   228,   229,   232,   233,   234,   230,
   231,   225,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   236,   241,   240,   242,     0,
   253,     0,   276,   269,     0,    25,    26,    27,    28,    29,
    30,    31,    32,    33,    34,     0,    49,    50,    35,    36,
    37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
    47,    48,     0,   100,   100,   295,     0,     0,   284,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   199,     0,   187,   188,   155,   156,     3,
     4,   153,   154,   157,   148,   149,   152,     0,     0,     0,
     0,   151,   150,   185,   186,   111,   111,   137,   212,   206,
   209,   210,     0,     0,   130,   226,     0,     0,     0,     0,
     0,     0,     0,     0,   171,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   291,     0,   293,
   288,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   200,     0,     0,   107,
   105,   143,     0,     0,   147,     0,   144,   131,   132,   205,
   207,     0,   103,   139,     0,     0,     0,   288,     0,     0,
     0,     0,     0,   235,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   298,     0,
     0,     0,   280,   281,     0,     0,     0,     0,     0,   277,
   278,     0,   296,     0,   102,   108,   106,   142,   145,   146,
   211,   208,   104,    98,     0,     0,     0,     0,     0,     0,
     0,     0,   170,     0,     0,     0,     0,     0,     0,     0,
   268,     0,     0,   100,   101,   100,   265,   287,     0,     0,
     0,     0,     0,   271,   272,   273,   268,     0,     0,   214,
   237,     0,     0,   159,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   267,     0,     0,     0,     0,
   292,   294,     0,     0,     0,   279,   282,   283,     0,   297,
    99,     0,     0,     0,   167,     0,     0,   161,   162,   163,
   166,   158,     0,   256,     0,     0,     0,   274,   275,   266,
   263,     0,   285,   164,   165,     0,     0,     0,   254,     0,
   255,     0,     0,   264,   160,   168,   169,     0,     0,     0,
     0,     0,     0,   261,     0,     0,   260,     0,   257,     0,
     0,     0
};

static const short yydefgoto[] = {    81,
   332,   261,   262,   263,   286,   303,   264,   265,   230,   231,
   145,   232,    22,    13,    36,   500,   378,   400,   464,   326,
   401,    82,    83,   233,    85,    86,   126,   243,   365,   266,
   366,   116,   580,     1,    55,     2,    59,   212,    46,   111,
   216,    87,   411,   351,   352,   353,    37,    91,    14,    94,
    15,    27,    16,   357,   267,    88,   269,   487,    39,    40,
    41,   103,   104,   546,   105,   309,   516,   517,   205,   206,
   439,   207,   208
};

static const short yypact[] = {-32768,
     9,   905,-32768,    63,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,   -34,    34,   -11,-32768,-32768,-32768,-32768,   -28,   -81,
    32,   127,   -23,-32768,-32768,-32768,    34,    87,   105,-32768,
-32768,-32768,-32768,-32768,-32768,   801,   -17,-32768,   -19,-32768,
    47,   -16,    25,    58,    59,-32768,    74,    87,   801,    11,
    11,-32768,-32768,    11,    11,-32768,-32768,-32768,-32768,-32768,
    61,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,   217,   218,   219,   518,
-32768,   104,    84,-32768,-32768,   -72,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,   839,     8,   253,-32768,-32768,
-32768,  1273,-32768,-32768,-32768,   210,    90,   224,   214,   215,
-32768,-32768,-32768,-32768,-32768,   868,   868,   898,   868,-32768,
    91,    92,-32768,-32768,   -72,   -74,    97,   213,-32768,    61,
  1071,-32768,  1071,  1071,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,  1071,-32768,   868,-32768,   192,    34,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
   293,    81,   868,   868,   868,   868,   868,   868,-32768,-32768,
-32768,-32768,   868,   868,   868,   868,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   868,
   868,   868,   868,   868,    34,-32768,    30,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,   -40,   102,   102,   135,   145,
   248,   161,   250,   163,   251,   165,   255,   256,   258,   188,
   262,   260,   982,   102,   102,   868,   868,   868,-32768,   604,
-32768,   122,   132,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,   225,   293,    81,   129,   134,   138,   139,   140,   898,
   141,   143,   147,   148,   149,-32768,-32768,-32768,-32768,   153,
-32768,   154,-32768,-32768,   801,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,   868,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   868,   155,   156,-32768,  1071,   126,   157,   158,
   159,   160,   166,   167,  1071,  1071,  1071,   168,   274,   801,
   868,   868,   287,-32768,   -18,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,   679,   898,   549,
   291,-32768,-32768,-32768,-32768,   -49,   -27,   -72,-32768,   104,
-32768,   171,   169,   709,-32768,-32768,   295,   174,   175,   898,
   898,   898,   898,   898,-32768,  -110,   898,   898,   898,   898,
   898,   315,   317,  1071,  1071,  1071,     1,-32768,    12,-32768,
   184,  1071,   182,   868,   868,   868,   868,   868,   187,   189,
   190,   868,   868,  1071,  1071,   194,-32768,   310,   329,-32768,
   102,-32768,   -39,   -68,-32768,   -73,-32768,-32768,-32768,-32768,
-32768,   763,   316,-32768,   203,   898,   898,   184,   206,   207,
   208,   209,   898,-32768,   211,   212,   216,   221,   323,  1071,
  1071,   226,   230,   231,  1071,   347,  1071,   868,-32768,   232,
  1071,   234,-32768,-32768,   238,   239,  1071,  1071,  1071,-32768,
-32768,   247,-32768,   868,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,   335,   337,   241,   254,   252,   898,   898,
   898,   898,-32768,   898,   898,   898,   898,   868,   257,   259,
   868,  1071,  1071,   264,-32768,   264,-32768,   265,  1071,   272,
   868,   868,   868,-32768,-32768,-32768,   868,  1071,   390,-32768,
-32768,   898,   898,-32768,   273,   271,   275,   276,   277,   278,
   279,   281,   282,   395,    26,   265,   285,   286,   288,   373,
-32768,-32768,   868,   289,  1071,-32768,-32768,-32768,   294,-32768,
-32768,   297,   298,   898,-32768,   898,   898,-32768,-32768,-32768,
-32768,-32768,  1071,-32768,  1160,    56,   392,-32768,-32768,-32768,
-32768,   290,-32768,-32768,-32768,   299,   303,   304,-32768,   292,
-32768,  1160,   431,-32768,-32768,-32768,-32768,   445,   311,  1071,
  1071,   447,   130,-32768,  1071,   449,-32768,  1071,-32768,   468,
   469,-32768
};

static const short yypgoto[] = {-32768,
-32768,   368,   370,   371,   222,   223,   374,   376,   -96,   -95,
  -498,-32768,   438,   458,  -134,-32768,  -298,    68,-32768,  -214,
-32768,   -44,-32768,   -36,-32768,   -79,   354,-32768,   128,   261,
  -187,    76,-32768,-32768,-32768,-32768,   436,-32768,-32768,-32768,
-32768,     4,-32768,    73,-32768,-32768,   459,-32768,-32768,-32768,
-32768,-32768,   491,-32768,  -469,  -104,   -60,   -88,-32768,   461,
-32768,   -71,-32768,-32768,-32768,-32768,    65,     7,-32768,-32768,
    83,-32768,-32768
};


#define	YYLAST		1423


static const short yytable[] = {    84,
   125,   143,   144,   327,   113,    23,   380,   132,  -174,    38,
   435,   133,    84,    92,    89,   275,   545,    38,   398,   344,
   345,   437,   134,    42,    43,    44,   268,    28,   268,   268,
   135,   136,   137,   138,   139,   140,   141,   142,   399,     3,
   268,   423,     4,    45,    23,   114,   115,   562,   125,   424,
     5,     6,     7,     8,     9,    10,    11,   436,   273,   131,
   135,   136,   137,   138,   139,   140,   141,   142,   436,    47,
   320,    12,   270,   271,    17,   560,    18,   238,   423,   217,
   218,  -111,   235,   423,   272,   127,   306,   239,   460,   310,
   129,   459,   569,    48,   311,   312,   313,   314,    29,    30,
    31,    32,    33,    34,    35,   287,   288,    25,   408,    26,
   131,   323,   423,   129,   318,   319,    58,   324,   458,    96,
    97,    98,    60,    99,   100,   101,   117,    56,   343,   118,
   119,    17,   409,    18,   106,   129,   304,   305,   131,   307,
   308,   131,    93,    90,   210,   211,   131,   131,   131,   131,
   403,   404,   406,    49,   321,   322,   346,   347,   348,   328,
   329,   -71,   -71,   315,   316,   317,   131,   131,     5,     6,
     7,    50,     9,    51,    11,   107,    52,   -70,   -70,   -69,
   -69,   -68,   -68,   544,    53,   521,   457,   522,   289,   290,
   291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
   301,   302,   268,   350,   330,   331,    99,   100,   108,   109,
   268,   268,   268,   561,  -112,   123,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,   110,
   374,    75,    76,   396,   120,   121,   122,   128,    84,   209,
    17,   213,    18,   214,   215,   234,   381,   236,   237,   375,
   240,   274,   241,   325,   389,   390,   391,   135,   136,   137,
   138,   139,   140,   141,   142,   -75,   376,   -74,   -73,   268,
   268,   268,   -72,   354,   348,   394,   -78,   268,   -79,   333,
   334,   382,   360,    84,   395,   131,   355,   361,   356,   268,
   268,   362,   363,   364,   367,   442,   368,   444,   445,   446,
   369,   370,   371,   450,   372,   373,   377,   379,   383,   384,
   385,   386,   393,   432,   433,   434,   397,   387,   388,   392,
   407,   440,   412,   413,   415,   268,   268,   416,   417,   430,
   268,   431,   268,   452,   453,   438,   268,   441,   447,   455,
   448,   449,   268,   268,   268,   454,   456,   131,   443,   131,
   131,   131,   398,   410,   465,   131,   451,   469,   470,   471,
   472,   478,   474,   475,   485,    77,   501,   476,    78,   479,
   480,    79,   477,    80,   484,   350,   486,   268,   268,   481,
   490,   482,   483,   489,   268,   491,   494,   495,   496,   492,
   493,   499,   502,   268,   276,   277,   278,   279,   280,   281,
   497,   131,   526,   527,   528,   503,   504,   531,   514,   543,
   282,   283,   284,   285,   515,   520,   523,   498,   143,   144,
   268,   518,   519,   525,   534,   535,   536,   537,   524,   436,
   563,   538,   539,   540,   550,   541,   542,   530,   268,   547,
   548,   513,   549,   568,   131,   570,   551,   564,   553,   143,
   144,   554,   555,   565,   131,   131,   131,   566,   567,   571,
   131,   575,   572,   578,   552,   268,   268,   581,   582,   200,
   268,   201,   202,   268,   358,   203,   359,   204,   102,    54,
   463,   242,   559,   112,   462,    57,   131,   418,   419,   420,
   421,   422,    24,   342,   425,   426,   427,   428,   429,    95,
   468,   576,   488,   529,     0,     0,     0,     0,     0,   573,
   574,     0,     0,     0,   577,     0,     0,   579,     0,     0,
   123,    62,    63,    64,    65,    66,    67,    68,    69,    70,
    71,    72,    73,    74,     0,     0,    75,    76,     0,     0,
     0,     0,     0,   466,   467,    17,     0,    18,     0,     0,
   473,   123,   219,   220,   221,   222,   223,   224,   225,   226,
   227,   228,   229,    73,    74,     0,     0,    75,    76,     0,
     0,     0,     0,     0,     0,     0,    17,     0,    18,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   505,   506,   507,   508,
     0,   509,   510,   511,   512,     0,   123,    62,    63,    64,
    65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
     0,     0,    75,    76,     0,     0,     0,     0,     0,   532,
   533,    17,     0,    18,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   349,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   556,     0,   557,   558,     0,     0,     0,     0,     0,
    77,     0,     0,    78,     0,     0,    79,     0,    80,   124,
     0,   123,   219,   220,   221,   222,   223,   224,   225,   226,
   227,   228,   229,    73,    74,     0,     0,    75,    76,     0,
     0,    77,     0,     0,    78,     0,    17,    79,    18,    80,
   405,   123,    62,    63,    64,    65,    66,    67,    68,    69,
    70,    71,    72,    73,    74,     0,     0,    75,    76,     0,
     0,     0,     0,     0,     0,     0,    17,     0,    18,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   414,     0,
     0,     0,     0,     0,     0,     0,    77,     0,     0,    78,
     0,     0,    79,     0,    80,   123,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,     0,
     0,    75,    76,     0,     0,     0,     0,     0,     0,     0,
    17,     0,    18,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   461,    61,    62,    63,    64,    65,    66,    67,
    68,    69,    70,    71,    72,    73,    74,     0,     0,    75,
    76,     0,     0,     0,     0,     0,     0,     0,    17,     0,
    18,    77,     0,     0,    78,     0,   402,    79,     0,    80,
     0,   130,    62,    63,    64,    65,    66,    67,    68,    69,
    70,    71,    72,    73,    74,     0,     0,    75,    76,     0,
     0,    77,     0,     0,    78,     0,    17,    79,    18,    80,
   123,    62,    63,    64,    65,    66,    67,    68,    69,    70,
    71,    72,    73,    74,     0,     0,    75,    76,     0,     0,
     0,     0,     0,     0,     0,    17,     0,    18,     0,     0,
   123,   219,   220,   221,   222,   223,   224,   225,   226,   227,
   228,   229,    73,    74,     0,    77,    75,    76,    78,     0,
     0,    79,     0,    80,     0,    17,     0,    18,     0,     0,
     0,   -81,    17,     0,    18,     0,     0,     0,     4,   -81,
   -81,     0,     0,     0,     0,     0,   -81,   -81,   -81,   -81,
   -81,   -81,   -81,    77,   -81,    19,    78,     0,     0,    79,
     0,    80,   -81,    20,     0,     0,     0,    21,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    77,     0,     0,    78,     0,     0,    79,     0,    80,
    75,    76,     0,   335,   336,   337,     0,     0,     0,    17,
     0,    18,     0,     0,     0,     0,     0,     0,     0,     0,
    77,     0,     0,    78,     0,     0,    79,     0,    80,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    77,     0,     0,    78,     0,     0,    79,     0,    80,     0,
     0,   151,   152,   153,   154,   155,   156,   157,   158,   159,
   160,   161,   162,   163,   164,   165,   166,   167,   168,   169,
   170,   253,   254,     0,     0,     0,     0,   244,   245,    75,
    76,   246,   247,   248,   249,   250,   251,     0,    17,     0,
    18,     0,     0,     0,     0,     0,     0,     0,   255,     0,
   256,   179,   180,   181,   182,     0,   257,   258,   259,   187,
   188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
   198,   199,   252,     0,     0,     0,     0,   338,     0,     0,
   339,     0,   340,     0,     0,   341,     0,     0,     0,     0,
   151,   152,   153,   154,   155,   156,   157,   158,   159,   160,
   161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
   253,   254,     0,     0,     0,     0,   244,   245,     0,     0,
   246,   247,   248,   249,   250,   251,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   255,     0,   256,
   179,   180,   181,   182,     0,   257,   258,   259,   187,   188,
   189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
   199,   252,     0,     0,     0,     0,     0,     0,     0,   260,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   151,
   152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
   162,   163,   164,   165,   166,   167,   168,   169,   170,   253,
   254,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   255,     0,   256,   179,
   180,   181,   182,     0,   257,   258,   259,   187,   188,   189,
   190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
   146,     0,     0,     0,     0,     0,     0,     0,   260,     0,
     0,   147,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   148,   149,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   150,     0,
     0,     0,   151,   152,   153,   154,   155,   156,   157,   158,
   159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
   169,   170,   171,   172,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   173,   174,   175,     0,     0,   176,
   177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
   187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
   197,   198,   199
};

static const short yycheck[] = {    36,
    80,    98,    98,   218,    49,     2,   305,    96,     0,    29,
    10,     4,    49,    33,    32,   150,   515,    29,    37,   234,
   235,    10,    15,    52,    53,    54,   131,    62,   133,   134,
     5,     6,     7,     8,     9,    10,    11,    12,    57,    31,
   145,   152,    34,    72,    41,    35,    36,   546,   128,   160,
    42,    43,    44,    45,    46,    47,    48,    57,   147,    96,
     5,     6,     7,     8,     9,    10,    11,    12,    57,   151,
   205,    63,   133,   134,    28,   545,    30,   152,   152,   116,
   117,   154,   119,   152,   145,    82,   175,   162,   162,   178,
   163,   160,   562,    62,   183,   184,   185,   186,    65,    66,
    67,    68,    69,    70,    71,    25,    26,    45,   158,    47,
   147,   152,   152,   163,   203,   204,    30,   158,   158,    73,
    74,    75,    18,    77,    78,    79,    51,   151,   233,    54,
    55,    28,   160,    30,   151,   163,   173,   174,   175,   176,
   177,   178,   162,   161,    55,    56,   183,   184,   185,   186,
   338,   339,   340,    27,   125,   126,   236,   237,   238,    25,
    26,    17,    18,   200,   201,   202,   203,   204,    42,    43,
    44,    45,    46,    47,    48,   151,    50,    17,    18,    17,
    18,    17,    18,   158,    58,   484,   401,   486,   108,   109,
   110,   111,   112,   113,   114,   115,   116,   117,   118,   119,
   120,   121,   307,   240,    17,    18,    77,    78,   151,   151,
   315,   316,   317,   158,   154,     3,     4,     5,     6,     7,
     8,     9,    10,    11,    12,    13,    14,    15,    16,   156,
   275,    19,    20,   322,    18,    18,    18,   154,   275,    30,
    28,    18,    30,    30,    30,   118,   307,   157,   157,   286,
   154,    60,    40,   152,   315,   316,   317,     5,     6,     7,
     8,     9,    10,    11,    12,    18,   303,    18,    18,   374,
   375,   376,    18,   152,   354,   320,    21,   382,    21,    18,
    21,   156,   154,   320,   321,   322,   155,   154,    64,   394,
   395,   154,   154,   154,   154,   384,   154,   386,   387,   388,
   154,   154,   154,   392,   152,   152,   152,   152,   152,   152,
   152,   152,    39,   374,   375,   376,    30,   152,   152,   152,
    30,   382,   152,   155,    30,   430,   431,   154,   154,    15,
   435,    15,   437,   394,   395,   152,   441,   156,   152,    30,
   152,   152,   447,   448,   449,   152,    18,   384,   385,   386,
   387,   388,    37,   350,   152,   392,   393,   152,   152,   152,
   152,    39,   152,   152,    18,   153,    30,   152,   156,   430,
   431,   159,   152,   161,   435,   412,   437,   482,   483,   154,
   441,   152,   152,   152,   489,   152,   447,   448,   449,   152,
   152,    57,   152,   498,   102,   103,   104,   105,   106,   107,
   154,   438,   491,   492,   493,   152,   155,    18,   152,    15,
   118,   119,   120,   121,   156,   152,   152,   454,   515,   515,
   525,   482,   483,   152,   152,   155,   152,   152,   489,    57,
    39,   155,   155,   155,   523,   155,   155,   498,   543,   155,
   155,   478,   155,   152,   481,    15,   158,   158,   155,   546,
   546,   155,   155,   155,   491,   492,   493,   155,   155,    15,
   497,    15,   152,    15,   525,   570,   571,     0,     0,   102,
   575,   102,   102,   578,   253,   102,   254,   102,    41,    22,
   413,   128,   543,    48,   412,    27,   523,   360,   361,   362,
   363,   364,     2,   233,   367,   368,   369,   370,   371,    39,
   418,   573,   438,   497,    -1,    -1,    -1,    -1,    -1,   570,
   571,    -1,    -1,    -1,   575,    -1,    -1,   578,    -1,    -1,
     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,
    -1,    -1,    -1,   416,   417,    28,    -1,    30,    -1,    -1,
   423,     3,     4,     5,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   469,   470,   471,   472,
    -1,   474,   475,   476,   477,    -1,     3,     4,     5,     6,
     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,   502,
   503,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    40,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   534,    -1,   536,   537,    -1,    -1,    -1,    -1,    -1,
   153,    -1,    -1,   156,    -1,    -1,   159,    -1,   161,   162,
    -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
    -1,   153,    -1,    -1,   156,    -1,    28,   159,    30,   161,
   162,     3,     4,     5,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,   156,
    -1,    -1,   159,    -1,   161,     3,     4,     5,     6,     7,
     8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    40,     3,     4,     5,     6,     7,     8,     9,
    10,    11,    12,    13,    14,    15,    16,    -1,    -1,    19,
    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,
    30,   153,    -1,    -1,   156,    -1,   158,   159,    -1,   161,
    -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
    -1,   153,    -1,    -1,   156,    -1,    28,   159,    30,   161,
     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,    -1,
     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
    13,    14,    15,    16,    -1,   153,    19,    20,   156,    -1,
    -1,   159,    -1,   161,    -1,    28,    -1,    30,    -1,    -1,
    -1,    27,    28,    -1,    30,    -1,    -1,    -1,    34,    35,
    36,    -1,    -1,    -1,    -1,    -1,    42,    43,    44,    45,
    46,    47,    48,   153,    50,    51,   156,    -1,    -1,   159,
    -1,   161,    58,    59,    -1,    -1,    -1,    63,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   153,    -1,    -1,   156,    -1,    -1,   159,    -1,   161,
    19,    20,    -1,    22,    23,    24,    -1,    -1,    -1,    28,
    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   153,    -1,    -1,   156,    -1,    -1,   159,    -1,   161,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   153,    -1,    -1,   156,    -1,    -1,   159,    -1,   161,    -1,
    -1,    80,    81,    82,    83,    84,    85,    86,    87,    88,
    89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
    99,   100,   101,    -1,    -1,    -1,    -1,    17,    18,    19,
    20,    21,    22,    23,    24,    25,    26,    -1,    28,    -1,
    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,    -1,
   129,   130,   131,   132,   133,    -1,   135,   136,   137,   138,
   139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
   149,   150,    62,    -1,    -1,    -1,    -1,   156,    -1,    -1,
   159,    -1,   161,    -1,    -1,   164,    -1,    -1,    -1,    -1,
    80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
    90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
   100,   101,    -1,    -1,    -1,    -1,    17,    18,    -1,    -1,
    21,    22,    23,    24,    25,    26,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,    -1,   129,
   130,   131,   132,   133,    -1,   135,   136,   137,   138,   139,
   140,   141,   142,   143,   144,   145,   146,   147,   148,   149,
   150,    62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   159,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
    81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
    91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
   101,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   127,    -1,   129,   130,
   131,   132,   133,    -1,   135,   136,   137,   138,   139,   140,
   141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
    38,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   159,    -1,
    -1,    49,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    60,    61,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,
    -1,    -1,    80,    81,    82,    83,    84,    85,    86,    87,
    88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
    98,    99,   100,   101,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   122,   123,   124,    -1,    -1,   127,
   128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
   138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
   148,   149,   150
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

case 80:
#line 358 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 81:
#line 361 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = new std::string(""); 
  ;
    break;}
case 89:
#line 368 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(""); ;
    break;}
case 96:
#line 373 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    *yyvsp[-1].String += *yyvsp[0].String; 
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String; 
    ;
    break;}
case 97:
#line 378 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(""); ;
    break;}
case 98:
#line 383 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 99:
#line 384 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ *yyvsp[-1].String += " " + *yyvsp[0].String; delete yyvsp[0].String; yyval.String = yyvsp[-1].String; ;
    break;}
case 100:
#line 387 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 101:
#line 388 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyvsp[-1].String->insert(0, ", "); 
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 102:
#line 396 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 103:
#line 402 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 105:
#line 406 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 106:
#line 407 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
      yyvsp[-1].String->insert(0, ", ");
      if (!yyvsp[0].String->empty())
        *yyvsp[-1].String += " " + *yyvsp[0].String;
      delete yyvsp[0].String;
      yyval.String = yyvsp[-1].String;
    ;
    break;}
case 108:
#line 417 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
      *yyvsp[-1].String += " " + *yyvsp[0].String;
      delete yyvsp[0].String;
      yyval.String = yyvsp[-1].String;
    ;
    break;}
case 126:
#line 439 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.Type.newTy = yyvsp[0].String; 
    yyval.Type.oldTy = OpaqueTy; 
  ;
    break;}
case 127:
#line 443 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.Type.newTy = yyvsp[0].String;
    yyval.Type.oldTy = UnresolvedTy;
  ;
    break;}
case 128:
#line 447 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.Type = yyvsp[0].Type; 
  ;
    break;}
case 129:
#line 450 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                   // Type UpReference
    yyvsp[0].String->insert(0, "\\");
    yyval.Type.newTy = yyvsp[0].String;
    yyval.Type.oldTy = NumericTy;
  ;
    break;}
case 130:
#line 455 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{           // Function derived type?
    *yyvsp[-3].Type.newTy += "( " + *yyvsp[-1].String + " )";
    delete yyvsp[-1].String;
    yyval.Type.newTy = yyvsp[-3].Type.newTy;
    yyval.Type.oldTy = FunctionTy;
  ;
    break;}
case 131:
#line 461 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{          // Sized array type?
    yyvsp[-3].String->insert(0,"[ ");
    *yyvsp[-3].String += " x " + *yyvsp[-1].Type.newTy + " ]";
    delete yyvsp[-1].Type.newTy;
    yyval.Type.newTy = yyvsp[-3].String;
    yyval.Type.oldTy = ArrayTy;
    yyval.Type.elemTy = yyvsp[-1].Type.oldTy;
  ;
    break;}
case 132:
#line 469 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{          // Packed array type?
    yyvsp[-3].String->insert(0,"< ");
    *yyvsp[-3].String += " x " + *yyvsp[-1].Type.newTy + " >";
    delete yyvsp[-1].Type.newTy;
    yyval.Type.newTy = yyvsp[-3].String;
    yyval.Type.oldTy = PackedTy;
    yyval.Type.elemTy = yyvsp[-1].Type.oldTy;
  ;
    break;}
case 133:
#line 477 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                        // Structure type?
    yyvsp[-1].String->insert(0, "{ ");
    *yyvsp[-1].String += " }";
    yyval.Type.newTy = yyvsp[-1].String;
    yyval.Type.oldTy = StructTy;
  ;
    break;}
case 134:
#line 483 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                                  // Empty structure type?
    yyval.Type.newTy = new std::string("{}");
    yyval.Type.oldTy = StructTy;
  ;
    break;}
case 135:
#line 487 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                             // Pointer type?
    *yyvsp[-1].Type.newTy += '*';
    yyval.Type.elemTy = yyvsp[-1].Type.oldTy;
    yyvsp[-1].Type.oldTy = PointerTy;
    yyval.Type = yyvsp[-1].Type;
  ;
    break;}
case 136:
#line 498 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].Type.newTy;
  ;
    break;}
case 137:
#line 501 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].Type.newTy;
    delete yyvsp[0].Type.newTy;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 139:
#line 510 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", ...";
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 140:
#line 515 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 141:
#line 518 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = new std::string();
  ;
    break;}
case 142:
#line 528 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ // Nonempty unsized arr
    yyval.Const.type = yyvsp[-3].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-3].Type.newTy);
    *yyval.Const.cnst += " [ " + *yyvsp[-1].String + " ]";
    delete yyvsp[-1].String;
  ;
    break;}
case 143:
#line 534 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-2].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-2].Type.newTy);
    *yyval.Const.cnst += "[ ]";
  ;
    break;}
case 144:
#line 539 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-2].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-2].Type.newTy);
    *yyval.Const.cnst += " c" + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 145:
#line 545 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ // Nonempty unsized arr
    yyval.Const.type = yyvsp[-3].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-3].Type.newTy);
    *yyval.Const.cnst += " < " + *yyvsp[-1].String + " >";
    delete yyvsp[-1].String;
  ;
    break;}
case 146:
#line 551 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-3].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-3].Type.newTy);
    *yyval.Const.cnst += " { " + *yyvsp[-1].String + " }";
    delete yyvsp[-1].String;
  ;
    break;}
case 147:
#line 557 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-2].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-2].Type.newTy);
    *yyval.Const.cnst += " {}";
  ;
    break;}
case 148:
#line 562 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst +=  " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 149:
#line 568 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 150:
#line 574 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 151:
#line 580 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 152:
#line 586 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 153:
#line 592 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{      // integral constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 154:
#line 598 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{            // integral constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 155:
#line 604 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                      // Boolean constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 156:
#line 610 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                     // Boolean constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 157:
#line 616 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                   // Float & Double constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(*yyvsp[-1].Type.newTy);
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 158:
#line 624 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string source = *yyvsp[-3].Const.cnst;
    TypeInfo DstTy = yyvsp[-1].Type;
    ResolveType(DstTy);
    if (*yyvsp[-5].String == "cast") {
      // Call getCastUpgrade to upgrade the old cast
      yyval.String = new std::string(getCastUpgrade(source, yyvsp[-3].Const.type, yyvsp[-1].Type, true));
    } else {
      // Nothing to upgrade, just create the cast constant expr
      yyval.String = new std::string(*yyvsp[-5].String);
      *yyval.String += "( " + source + " to " + *yyvsp[-1].Type.newTy + ")";
    }
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); delete yyvsp[-2].String; yyvsp[-1].Type.destroy();
  ;
    break;}
case 159:
#line 638 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-4].String += "(" + *yyvsp[-2].Const.cnst;
    for (unsigned i = 0; i < yyvsp[-1].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[-1].ValList)[i];
      *yyvsp[-4].String += ", " + *VI.val;
      VI.destroy();
    }
    *yyvsp[-4].String += ")";
    yyval.String = yyvsp[-4].String;
    yyvsp[-2].Const.destroy();
    delete yyvsp[-1].ValList;
  ;
    break;}
case 160:
#line 650 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-7].String += "(" + *yyvsp[-5].Const.cnst + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-5].Const.destroy(); yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 161:
#line 655 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* op = getDivRemOpcode(*yyvsp[-5].String, yyvsp[-3].Const.type); 
    yyval.String = new std::string(op);
    *yyval.String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
  ;
    break;}
case 162:
#line 661 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 163:
#line 666 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
#if UPGRADE_SETCOND_OPS
    *yyvsp[-5].String = getCompareOp(*yyvsp[-5].String, yyvsp[-3].Const.type);
#endif
    *yyvsp[-5].String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 164:
#line 674 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += "(" + *yyvsp[-5].String + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 165:
#line 679 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += "(" + *yyvsp[-5].String + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 166:
#line 684 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* shiftop = yyvsp[-5].String->c_str();
    if (*yyvsp[-5].String == "shr")
      shiftop = (yyvsp[-3].Const.type.isUnsigned()) ? "lshr" : "ashr";
    yyval.String = new std::string(shiftop);
    *yyval.String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
  ;
    break;}
case 167:
#line 692 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 168:
#line 697 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-7].String += "(" + *yyvsp[-5].Const.cnst + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-5].Const.destroy(); yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 169:
#line 702 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-7].String += "(" + *yyvsp[-5].Const.cnst + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-5].Const.destroy(); yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 170:
#line 712 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].Const.cnst;
    yyvsp[0].Const.destroy();
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 171:
#line 717 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(*yyvsp[0].Const.cnst); yyvsp[0].Const.destroy(); ;
    break;}
case 174:
#line 732 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
;
    break;}
case 175:
#line 737 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 176:
#line 740 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << *yyvsp[0].String << "\n";
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 177:
#line 745 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "module asm " << " " << *yyvsp[0].String << "\n";
    yyval.String = 0;
  ;
    break;}
case 178:
#line 749 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "implementation\n";
    yyval.String = 0;
  ;
    break;}
case 179:
#line 753 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = 0; ;
    break;}
case 181:
#line 755 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = yyvsp[0].String; *yyval.String = "external"; ;
    break;}
case 182:
#line 758 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    EnumeratedTypes.push_back(yyvsp[0].Type);
    if (!yyvsp[-2].String->empty()) {
      NamedTypes[*yyvsp[-2].String].newTy = new std::string(*yyvsp[0].Type.newTy);
      NamedTypes[*yyvsp[-2].String].oldTy = yyvsp[0].Type.oldTy;
      NamedTypes[*yyvsp[-2].String].elemTy = yyvsp[0].Type.elemTy;
      *O << *yyvsp[-2].String << " = ";
    }
    *O << "type " << *yyvsp[0].Type.newTy << "\n";
    delete yyvsp[-2].String; delete yyvsp[-1].String; yyvsp[0].Type.destroy();
    yyval.String = 0;
  ;
    break;}
case 183:
#line 770 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{       // Function prototypes can be in const pool
    *O << *yyvsp[0].String << "\n";
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 184:
#line 775 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{  // Asm blocks can be in the const pool
    *O << *yyvsp[-2].String << " " << *yyvsp[-1].String << " " << *yyvsp[0].String << "\n";
    delete yyvsp[-2].String; delete yyvsp[-1].String; delete yyvsp[0].String; 
    yyval.String = 0;
  ;
    break;}
case 185:
#line 780 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      *O << *yyvsp[-4].String << " = ";
      Globals[*yyvsp[-4].String] = yyvsp[-1].Const.type.clone();
    }
    *O << *yyvsp[-3].String << " " << *yyvsp[-2].String << " " << *yyvsp[-1].Const.cnst << " " << *yyvsp[0].String << "\n";
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; yyvsp[-1].Const.destroy(); delete yyvsp[0].String; 
    yyval.String = 0;
  ;
    break;}
case 186:
#line 789 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      *O << *yyvsp[-4].String << " = ";
      Globals[*yyvsp[-4].String] = yyvsp[-1].Type.clone();
    }
    *O <<  *yyvsp[-3].String << " " << *yyvsp[-2].String << " " << *yyvsp[-1].Type.newTy << " " << *yyvsp[0].String << "\n";
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; yyvsp[-1].Type.destroy(); delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 187:
#line 798 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      *O << *yyvsp[-4].String << " = ";
      Globals[*yyvsp[-4].String] = yyvsp[-1].Type.clone();
    }
    *O << *yyvsp[-3].String << " " << *yyvsp[-2].String << " " << *yyvsp[-1].Type.newTy << " " << *yyvsp[0].String << "\n";
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; yyvsp[-1].Type.destroy(); delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 188:
#line 807 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      *O << *yyvsp[-4].String << " = ";
      Globals[*yyvsp[-4].String] = yyvsp[-1].Type.clone();
    }
    *O << *yyvsp[-3].String << " " << *yyvsp[-2].String << " " << *yyvsp[-1].Type.newTy << " " << *yyvsp[0].String << "\n";
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; yyvsp[-1].Type.destroy(); delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 189:
#line 816 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    *O << *yyvsp[-1].String << " " << *yyvsp[0].String << "\n";
    delete yyvsp[-1].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 190:
#line 821 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << *yyvsp[-2].String << " = " << *yyvsp[0].String << "\n";
    delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 191:
#line 826 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.String = 0;
  ;
    break;}
case 195:
#line 836 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 196:
#line 841 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    if (*yyvsp[0].String == "64")
      SizeOfPointer = 64;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 197:
#line 848 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 198:
#line 853 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 199:
#line 860 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyvsp[-1].String->insert(0, "[ ");
    *yyvsp[-1].String += " ]";
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 200:
#line 867 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 202:
#line 873 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = new std::string();
  ;
    break;}
case 206:
#line 882 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 207:
#line 884 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
  yyval.String = yyvsp[-1].Type.newTy;
  if (!yyvsp[0].String->empty())
    *yyval.String += " " + *yyvsp[0].String;
  delete yyvsp[0].String;
;
    break;}
case 208:
#line 891 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 209:
#line 895 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 210:
#line 899 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 211:
#line 902 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", ...";
    yyval.String = yyvsp[-2].String;
    delete yyvsp[0].String;
  ;
    break;}
case 212:
#line 907 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 213:
#line 910 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 214:
#line 913 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-7].String->empty()) {
      *yyvsp[-7].String += " ";
    }
    *yyvsp[-7].String += *yyvsp[-6].Type.newTy + " " + *yyvsp[-5].String + "(" + *yyvsp[-3].String + ")";
    if (!yyvsp[-1].String->empty()) {
      *yyvsp[-7].String += " " + *yyvsp[-1].String;
    }
    if (!yyvsp[0].String->empty()) {
      *yyvsp[-7].String += " " + *yyvsp[0].String;
    }
    yyvsp[-6].Type.destroy();
    delete yyvsp[-5].String;
    delete yyvsp[-3].String;
    delete yyvsp[-1].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 215:
#line 932 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string("{"); delete yyvsp[0].String; ;
    break;}
case 216:
#line 933 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string ("{"); ;
    break;}
case 217:
#line 935 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
  if (!yyvsp[-2].String->empty()) {
    *O << *yyvsp[-2].String << " ";
  }
  *O << *yyvsp[-1].String << " " << *yyvsp[0].String << "\n";
  delete yyvsp[-2].String; delete yyvsp[-1].String; delete yyvsp[0].String;
  yyval.String = 0;
;
    break;}
case 218:
#line 944 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string("}"); delete yyvsp[0].String; ;
    break;}
case 219:
#line 945 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string("}"); ;
    break;}
case 220:
#line 947 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
  if (yyvsp[-1].String)
    *O << *yyvsp[-1].String;
  *O << '\n' << *yyvsp[0].String << "\n";
  yyval.String = 0;
;
    break;}
case 221:
#line 955 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 224:
#line 961 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    if (!yyvsp[-1].String->empty())
      *yyvsp[-2].String += " " + *yyvsp[-1].String;
    *yyvsp[-2].String += " " + *yyvsp[0].String;
    delete yyvsp[-1].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 225:
#line 974 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 235:
#line 980 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyvsp[-1].String->insert(0, "<");
    *yyvsp[-1].String += ">";
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 237:
#line 986 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-3].String->empty()) {
      *yyvsp[-4].String += " " + *yyvsp[-3].String;
    }
    *yyvsp[-4].String += " " + *yyvsp[-2].String + ", " + *yyvsp[0].String;
    delete yyvsp[-3].String; delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = yyvsp[-4].String;
  ;
    break;}
case 240:
#line 999 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Value.val = yyvsp[0].String;
    yyval.Value.constant = false;
    yyval.Value.type.newTy = 0;
    yyval.Value.type.oldTy = UnresolvedTy;
  ;
    break;}
case 241:
#line 1005 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Value.val = yyvsp[0].String;
    yyval.Value.constant = true;
    yyval.Value.type.newTy = 0;
    yyval.Value.type.oldTy = UnresolvedTy;
  ;
    break;}
case 242:
#line 1016 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Value = yyvsp[0].Value;
    yyval.Value.type = yyvsp[-1].Type;
    yyval.Value.val->insert(0, *yyvsp[-1].Type.newTy + " ");
  ;
    break;}
case 243:
#line 1022 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 244:
#line 1025 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ // Do not allow functions with 0 basic blocks   
    yyval.String = 0;
  ;
    break;}
case 245:
#line 1033 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 246:
#line 1037 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[0].String << "\n";
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 247:
#line 1042 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 248:
#line 1045 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << *yyvsp[0].String << "\n";
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 250:
#line 1051 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = yyvsp[0].String; *yyval.String = "unwind"; ;
    break;}
case 251:
#line 1053 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{              // Return with a result...
    *O << "    " << *yyvsp[-1].String << " " << *yyvsp[0].Value.val << "\n";
    delete yyvsp[-1].String; yyvsp[0].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 252:
#line 1058 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                                       // Return with no result...
    *O << "    " << *yyvsp[-1].String << " " << *yyvsp[0].Type.newTy << "\n";
    delete yyvsp[-1].String; yyvsp[0].Type.destroy();
    yyval.String = 0;
  ;
    break;}
case 253:
#line 1063 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                         // Unconditional Branch...
    *O << "    " << *yyvsp[-2].String << " " << *yyvsp[-1].Type.newTy << " " << *yyvsp[0].Value.val << "\n";
    delete yyvsp[-2].String; yyvsp[-1].Type.destroy(); yyvsp[0].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 254:
#line 1068 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{  
    *O << "    " << *yyvsp[-8].String << " " << *yyvsp[-7].Type.newTy << " " << *yyvsp[-6].Value.val << ", " 
       << *yyvsp[-4].Type.newTy << " " << *yyvsp[-3].Value.val << ", " << *yyvsp[-1].Type.newTy << " " 
       << *yyvsp[0].Value.val << "\n";
    delete yyvsp[-8].String; yyvsp[-7].Type.destroy(); yyvsp[-6].Value.destroy(); yyvsp[-4].Type.destroy(); yyvsp[-3].Value.destroy(); 
    yyvsp[-1].Type.destroy(); yyvsp[0].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 255:
#line 1076 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[-8].String << " " << *yyvsp[-7].Type.newTy << " " << *yyvsp[-6].Value.val << ", " 
       << *yyvsp[-4].Type.newTy << " " << *yyvsp[-3].Value.val << " [" << *yyvsp[-1].String << " ]\n";
    delete yyvsp[-8].String; yyvsp[-7].Type.destroy(); yyvsp[-6].Value.destroy(); yyvsp[-4].Type.destroy(); yyvsp[-3].Value.destroy(); 
    delete yyvsp[-1].String;
    yyval.String = 0;
  ;
    break;}
case 256:
#line 1083 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[-7].String << " " << *yyvsp[-6].Type.newTy << " " << *yyvsp[-5].Value.val << ", " 
       << *yyvsp[-3].Type.newTy << " " << *yyvsp[-2].Value.val << "[]\n";
    delete yyvsp[-7].String; yyvsp[-6].Type.destroy(); yyvsp[-5].Value.destroy(); yyvsp[-3].Type.destroy(); yyvsp[-2].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 257:
#line 1090 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    ";
    if (!yyvsp[-13].String->empty())
      *O << *yyvsp[-13].String << " = ";
    *O << *yyvsp[-12].String << " " << *yyvsp[-11].String << " " << *yyvsp[-10].Type.newTy << " " << *yyvsp[-9].Value.val << " (";
    for (unsigned i = 0; i < yyvsp[-7].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[-7].ValList)[i];
      *O << *VI.val;
      if (i+1 < yyvsp[-7].ValList->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *yyvsp[-5].String << " " << *yyvsp[-4].Type.newTy << " " << *yyvsp[-3].Value.val << " " 
       << *yyvsp[-2].String << " " << *yyvsp[-1].Type.newTy << " " << *yyvsp[0].Value.val << "\n";
    delete yyvsp[-13].String; delete yyvsp[-12].String; delete yyvsp[-11].String; yyvsp[-10].Type.destroy(); yyvsp[-9].Value.destroy(); delete yyvsp[-7].ValList; 
    delete yyvsp[-5].String; yyvsp[-4].Type.destroy(); yyvsp[-3].Value.destroy(); delete yyvsp[-2].String; yyvsp[-1].Type.destroy(); 
    yyvsp[0].Value.destroy(); 
    yyval.String = 0;
  ;
    break;}
case 258:
#line 1109 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[0].String << "\n";
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 259:
#line 1114 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[0].String << "\n";
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 260:
#line 1120 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Type.newTy + " " + *yyvsp[-3].String + ", " + *yyvsp[-1].Type.newTy + " " + *yyvsp[0].Value.val;
    yyvsp[-4].Type.destroy(); delete yyvsp[-3].String; yyvsp[-1].Type.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 261:
#line 1125 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyvsp[-3].String->insert(0, *yyvsp[-4].Type.newTy + " " );
    *yyvsp[-3].String += ", " + *yyvsp[-1].Type.newTy + " " + *yyvsp[0].Value.val;
    yyvsp[-4].Type.destroy(); yyvsp[-1].Type.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-3].String;
  ;
    break;}
case 262:
#line 1133 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-1].String->empty())
      *yyvsp[-1].String += " = ";
    *yyvsp[-1].String += *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String; 
  ;
    break;}
case 263:
#line 1142 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{    // Used for PHI nodes
    yyvsp[-3].Value.val->insert(0, *yyvsp[-5].Type.newTy + "[");
    *yyvsp[-3].Value.val += "," + *yyvsp[-1].Value.val + "]";
    yyvsp[-5].Type.destroy(); yyvsp[-1].Value.destroy();
    yyval.String = new std::string(*yyvsp[-3].Value.val);
    yyvsp[-3].Value.destroy();
  ;
    break;}
case 264:
#line 1149 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += ", [" + *yyvsp[-3].Value.val + "," + *yyvsp[-1].Value.val + "]";
    yyvsp[-3].Value.destroy(); yyvsp[-1].Value.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 265:
#line 1157 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.ValList = new ValueList();
    yyval.ValList->push_back(yyvsp[0].Value);
  ;
    break;}
case 266:
#line 1161 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyvsp[-2].ValList->push_back(yyvsp[0].Value);
    yyval.ValList = yyvsp[-2].ValList;
  ;
    break;}
case 267:
#line 1168 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.ValList = yyvsp[0].ValList; ;
    break;}
case 268:
#line 1169 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.ValList = new ValueList(); ;
    break;}
case 269:
#line 1173 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 271:
#line 1181 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* op = getDivRemOpcode(*yyvsp[-4].String, yyvsp[-3].Type); 
    yyval.String = new std::string(op);
    *yyval.String += " " + *yyvsp[-3].Type.newTy + " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    delete yyvsp[-4].String; yyvsp[-3].Type.destroy(); yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 272:
#line 1187 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-4].String += " " + *yyvsp[-3].Type.newTy + " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyvsp[-3].Type.destroy(); yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-4].String;
  ;
    break;}
case 273:
#line 1192 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
#if UPGRADE_SETCOND_OPS
    *yyvsp[-4].String = getCompareOp(*yyvsp[-4].String, yyvsp[-3].Type);
#endif
    *yyvsp[-4].String += " " + *yyvsp[-3].Type.newTy + " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyvsp[-3].Type.destroy(); yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-4].String;
  ;
    break;}
case 274:
#line 1200 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += " " + *yyvsp[-5].String + " " + *yyvsp[-3].Value.val + "," + *yyvsp[-1].Value.val + ")";
    delete yyvsp[-5].String; yyvsp[-3].Value.destroy(); yyvsp[-1].Value.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 275:
#line 1205 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += " " + *yyvsp[-5].String + " " + *yyvsp[-3].Value.val + "," + *yyvsp[-1].Value.val + ")";
    delete yyvsp[-5].String; yyvsp[-3].Value.destroy(); yyvsp[-1].Value.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 276:
#line 1210 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].Value.val;
    yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 277:
#line 1215 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* shiftop = yyvsp[-3].String->c_str();
    if (*yyvsp[-3].String == "shr")
      shiftop = (yyvsp[-2].Value.type.isUnsigned()) ? "lshr" : "ashr";
    yyval.String = new std::string(shiftop);
    *yyval.String += " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    delete yyvsp[-3].String; yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 278:
#line 1223 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string source = *yyvsp[-2].Value.val;
    TypeInfo SrcTy = yyvsp[-2].Value.type;
    TypeInfo DstTy = yyvsp[0].Type;
    ResolveType(DstTy);
    yyval.String = new std::string();
    if (*yyvsp[-3].String == "cast") {
      *yyval.String +=  getCastUpgrade(source, SrcTy, DstTy, false);
    } else {
      *yyval.String += *yyvsp[-3].String + " " + source + " to " + *DstTy.newTy;
    }
    delete yyvsp[-3].String; yyvsp[-2].Value.destroy();
    delete yyvsp[-1].String; yyvsp[0].Type.destroy();
  ;
    break;}
case 279:
#line 1237 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Value.val + ", " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyvsp[-4].Value.destroy(); yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 280:
#line 1242 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-3].String += " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Type.newTy;
    yyvsp[-2].Value.destroy(); yyvsp[0].Type.destroy();
    yyval.String = yyvsp[-3].String;
  ;
    break;}
case 281:
#line 1247 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-3].String += " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-3].String;
  ;
    break;}
case 282:
#line 1252 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Value.val + ", " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyvsp[-4].Value.destroy(); yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 283:
#line 1257 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Value.val + ", " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyvsp[-4].Value.destroy(); yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 284:
#line 1262 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 285:
#line 1267 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-5].String->empty())
      *yyvsp[-6].String += " " + *yyvsp[-5].String;
    if (!yyvsp[-6].String->empty())
      *yyvsp[-6].String += " ";
    *yyvsp[-6].String += *yyvsp[-4].Type.newTy + " " + *yyvsp[-3].Value.val + "(";
    for (unsigned i = 0; i < yyvsp[-1].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[-1].ValList)[i];
      *yyvsp[-6].String += *VI.val;
      if (i+1 < yyvsp[-1].ValList->size())
        *yyvsp[-6].String += ", ";
      VI.destroy();
    }
    *yyvsp[-6].String += ")";
    delete yyvsp[-5].String; yyvsp[-4].Type.destroy(); yyvsp[-3].Value.destroy(); delete yyvsp[-1].ValList;
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 287:
#line 1289 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.ValList = yyvsp[0].ValList; ;
    break;}
case 288:
#line 1290 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{  yyval.ValList = new ValueList(); ;
    break;}
case 290:
#line 1295 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 291:
#line 1298 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " " + *yyvsp[-1].Type.newTy;
    if (!yyvsp[0].String->empty())
      *yyvsp[-2].String += " " + *yyvsp[0].String;
    yyvsp[-1].Type.destroy(); delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 292:
#line 1305 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Type.newTy + ", " + *yyvsp[-2].Type.newTy + " " + *yyvsp[-1].Value.val;
    if (!yyvsp[0].String->empty())
      *yyvsp[-5].String += " " + *yyvsp[0].String;
    yyvsp[-4].Type.destroy(); yyvsp[-2].Type.destroy(); yyvsp[-1].Value.destroy(); delete yyvsp[0].String;
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 293:
#line 1312 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " " + *yyvsp[-1].Type.newTy;
    if (!yyvsp[0].String->empty())
      *yyvsp[-2].String += " " + *yyvsp[0].String;
    yyvsp[-1].Type.destroy(); delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 294:
#line 1319 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Type.newTy + ", " + *yyvsp[-2].Type.newTy + " " + *yyvsp[-1].Value.val;
    if (!yyvsp[0].String->empty())
      *yyvsp[-5].String += " " + *yyvsp[0].String;
    yyvsp[-4].Type.destroy(); yyvsp[-2].Type.destroy(); yyvsp[-1].Value.destroy(); delete yyvsp[0].String;
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 295:
#line 1326 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].Value.val;
    yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 296:
#line 1331 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-3].String->empty())
      *yyvsp[-3].String += " ";
    *yyvsp[-3].String += *yyvsp[-2].String + " " + *yyvsp[-1].Type.newTy + " " + *yyvsp[0].Value.val;
    delete yyvsp[-2].String; yyvsp[-1].Type.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-3].String;
  ;
    break;}
case 297:
#line 1338 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-5].String->empty())
      *yyvsp[-5].String += " ";
    *yyvsp[-5].String += *yyvsp[-4].String + " " + *yyvsp[-3].Value.val + ", " + *yyvsp[-1].Type.newTy + " " + *yyvsp[0].Value.val;
    delete yyvsp[-4].String; yyvsp[-3].Value.destroy(); yyvsp[-1].Type.destroy(); yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 298:
#line 1345 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    // Upgrade the indices
    for (unsigned i = 0; i < yyvsp[0].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[0].ValList)[i];
      if (VI.type.isUnsigned() && !VI.isConstant() && 
          VI.type.getBitWidth() < 64) {
        std::string* old = VI.val;
        *O << "    %gep_upgrade" << unique << " = zext " << *old 
           << " to ulong\n";
        VI.val = new std::string("ulong %gep_upgrade" + llvm::utostr(unique++));
        VI.type.oldTy = ULongTy;
        delete old;
      }
    }
    *yyvsp[-3].String += " " + *yyvsp[-2].Type.newTy + " " + *yyvsp[-1].Value.val;
    for (unsigned i = 0; i < yyvsp[0].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[0].ValList)[i];
      *yyvsp[-3].String += ", " + *VI.val;
      VI.destroy();
    }
    yyvsp[-2].Type.destroy(); yyvsp[-1].Value.destroy(); delete yyvsp[0].ValList;
    yyval.String = yyvsp[-3].String;
  ;
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
#line 1369 "/Volumes/Big2/llvm/llvm/tools/llvm-upgrade/UpgradeParser.y"


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
