
/*  A Bison parser, made from /Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y
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

#line 14 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"

#include "ParserInternals.h"
#include <algorithm>
#include <map>
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
static uint64_t unique = 1;

// This bool controls whether attributes are ever added to function declarations
// definitions and calls.
static bool AddAttributes = false;

// This bool is used to communicate between the InstVal and Inst rules about
// whether or not a cast should be deleted. When the flag is set, InstVal has
// determined that the cast is a candidate. However, it can only be deleted if
// the value being casted is the same value name as the instruction. The Inst
// rule makes that comparison if the flag is set and comments out the
// instruction if they match.
static bool deleteUselessCastFlag = false;
static std::string* deleteUselessCastName = 0;

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
                     std::ostream &out, bool debug, bool addAttrs)
{
  Upgradelineno = 1; 
  CurFilename = infile;
  LexInput = &in;
  yydebug = debug;
  AddAttributes = addAttrs;
  O = &out;

  if (yyparse()) {
    std::cerr << "Parse failed.\n";
    out << "llvm-upgrade parse failed.\n";
    exit(1);
  }
}

TypeInfo* ResolveType(TypeInfo*& Ty) {
  if (Ty->isUnresolved()) {
    if (Ty->getNewTy()[0] == '%' && isdigit(Ty->getNewTy()[1])) {
      unsigned ref = atoi(&((Ty->getNewTy().c_str())[1])); // skip the %
      if (ref < EnumeratedTypes.size()) {
        Ty = &EnumeratedTypes[ref];
        return Ty;
      } else {
        std::string msg("Can't resolve numbered type: ");
        msg += Ty->getNewTy();
        yyerror(msg.c_str());
      }
    } else {
      TypeMap::iterator I = NamedTypes.find(Ty->getNewTy());
      if (I != NamedTypes.end()) {
        Ty = &I->second;
        return Ty;
      } else {
        std::string msg("Cannot resolve type: ");
        msg += Ty->getNewTy();
        yyerror(msg.c_str());
      }
    }
  }
  // otherwise its already resolved.
  return Ty;
}

static const char* getCastOpcode(
  std::string& Source, const TypeInfo* SrcTy, const TypeInfo* DstTy) 
{
  unsigned SrcBits = SrcTy->getBitWidth();
  unsigned DstBits = DstTy->getBitWidth();
  const char* opcode = "bitcast";
  // Run through the possibilities ...
  if (DstTy->isIntegral()) {                        // Casting to integral
    if (SrcTy->isIntegral()) {                      // Casting from integral
      if (DstBits < SrcBits)
        opcode = "trunc";
      else if (DstBits > SrcBits) {                // its an extension
        if (SrcTy->isSigned())
          opcode ="sext";                          // signed -> SEXT
        else
          opcode = "zext";                         // unsigned -> ZEXT
      } else {
        opcode = "bitcast";                        // Same size, No-op cast
      }
    } else if (SrcTy->isFloatingPoint()) {          // Casting from floating pt
      if (DstTy->isSigned()) 
        opcode = "fptosi";                         // FP -> sint
      else
        opcode = "fptoui";                         // FP -> uint 
    } else if (SrcTy->isPacked()) {
      assert(DstBits == SrcTy->getBitWidth() &&
               "Casting packed to integer of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(SrcTy->isPointer() &&
             "Casting from a value that is not first-class type");
      opcode = "ptrtoint";                         // ptr -> int
    }
  } else if (DstTy->isFloatingPoint()) {           // Casting to floating pt
    if (SrcTy->isIntegral()) {                     // Casting from integral
      if (SrcTy->isSigned())
        opcode = "sitofp";                         // sint -> FP
      else
        opcode = "uitofp";                         // uint -> FP
    } else if (SrcTy->isFloatingPoint()) {         // Casting from floating pt
      if (DstBits < SrcBits) {
        opcode = "fptrunc";                        // FP -> smaller FP
      } else if (DstBits > SrcBits) {
        opcode = "fpext";                          // FP -> larger FP
      } else  {
        opcode ="bitcast";                         // same size, no-op cast
      }
    } else if (SrcTy->isPacked()) {
      assert(DstBits == SrcTy->getBitWidth() &&
             "Casting packed to floating point of different width");
        opcode = "bitcast";                        // same size, no-op cast
    } else {
      assert(0 && "Casting pointer or non-first class to float");
    }
  } else if (DstTy->isPacked()) {
    if (SrcTy->isPacked()) {
      assert(DstTy->getBitWidth() == SrcTy->getBitWidth() &&
             "Casting packed to packed of different widths");
      opcode = "bitcast";                          // packed -> packed
    } else if (DstTy->getBitWidth() == SrcBits) {
      opcode = "bitcast";                          // float/int -> packed
    } else {
      assert(!"Illegal cast to packed (wrong type or size)");
    }
  } else if (DstTy->isPointer()) {
    if (SrcTy->isPointer()) {
      opcode = "bitcast";                          // ptr -> ptr
    } else if (SrcTy->isIntegral()) {
      opcode = "inttoptr";                         // int -> ptr
    } else {
      assert(!"Casting invalid type to pointer");
    }
  } else {
    assert(!"Casting to type that is not first-class");
  }
  return opcode;
}

static std::string getCastUpgrade(const std::string& Src, TypeInfo* SrcTy,
                                  TypeInfo* DstTy, bool isConst)
{
  std::string Result;
  std::string Source = Src;
  if (SrcTy->isFloatingPoint() && DstTy->isPointer()) {
    // fp -> ptr cast is no longer supported but we must upgrade this
    // by doing a double cast: fp -> int -> ptr
    if (isConst)
      Source = "i64 fptoui(" + Source + " to i64)";
    else {
      *O << "    %cast_upgrade" << unique << " = fptoui " << Source 
         << " to i64\n";
      Source = "i64 %cast_upgrade" + llvm::utostr(unique);
    }
    // Update the SrcTy for the getCastOpcode call below
    delete SrcTy;
    SrcTy = new TypeInfo("i64", ULongTy);
  } else if (DstTy->isBool()) {
    // cast type %x to bool was previously defined as setne type %x, null
    // The cast semantic is now to truncate, not compare so we must retain
    // the original intent by replacing the cast with a setne
    const char* comparator = SrcTy->isPointer() ? ", null" : 
      (SrcTy->isFloatingPoint() ? ", 0.0" : 
       (SrcTy->isBool() ? ", false" : ", 0"));
    const char* compareOp = SrcTy->isFloatingPoint() ? "fcmp one " : "icmp ne ";
    if (isConst) { 
      Result = "(" + Source + comparator + ")";
      Result = compareOp + Result;
    } else
      Result = compareOp + Source + comparator;
    return Result; // skip cast processing below
  }
  ResolveType(SrcTy);
  ResolveType(DstTy);
  std::string Opcode(getCastOpcode(Source, SrcTy, DstTy));
  if (isConst)
    Result += Opcode + "( " + Source + " to " + DstTy->getNewTy() + ")";
  else
    Result += Opcode + " " + Source + " to " + DstTy->getNewTy();
  return Result;
}

const char* getDivRemOpcode(const std::string& opcode, TypeInfo* TI) {
  const char* op = opcode.c_str();
  const TypeInfo* Ty = ResolveType(TI);
  if (Ty->isPacked())
    Ty = Ty->getElementType();
  if (opcode == "div")
    if (Ty->isFloatingPoint())
      op = "fdiv";
    else if (Ty->isUnsigned())
      op = "udiv";
    else if (Ty->isSigned())
      op = "sdiv";
    else
      yyerror("Invalid type for div instruction");
  else if (opcode == "rem")
    if (Ty->isFloatingPoint())
      op = "frem";
    else if (Ty->isUnsigned())
      op = "urem";
    else if (Ty->isSigned())
      op = "srem";
    else
      yyerror("Invalid type for rem instruction");
  return op;
}

std::string 
getCompareOp(const std::string& setcc, const TypeInfo* TI) {
  assert(setcc.length() == 5);
  char cc1 = setcc[3];
  char cc2 = setcc[4];
  assert(cc1 == 'e' || cc1 == 'n' || cc1 == 'l' || cc1 == 'g');
  assert(cc2 == 'q' || cc2 == 'e' || cc2 == 'e' || cc2 == 't');
  std::string result("xcmp xxx");
  result[6] = cc1;
  result[7] = cc2;
  if (TI->isFloatingPoint()) {
    result[0] = 'f';
    result[5] = 'o';
    if (cc1 == 'n')
      result[5] = 'u'; // NE maps to unordered
    else
      result[5] = 'o'; // everything else maps to ordered
  } else if (TI->isIntegral() || TI->isPointer()) {
    result[0] = 'i';
    if ((cc1 == 'e' && cc2 == 'q') || (cc1 == 'n' && cc2 == 'e'))
      result.erase(5,1);
    else if (TI->isSigned())
      result[5] = 's';
    else if (TI->isUnsigned() || TI->isPointer() || TI->isBool())
      result[5] = 'u';
    else
      yyerror("Invalid integral type for setcc");
  }
  return result;
}

static TypeInfo* getFunctionReturnType(TypeInfo* PFTy) {
  ResolveType(PFTy);
  if (PFTy->isPointer()) {
    TypeInfo* ElemTy = PFTy->getElementType();
    ResolveType(ElemTy);
    if (ElemTy->isFunction())
      return ElemTy->getResultType()->clone();
  } else if (PFTy->isFunction()) {
    return PFTy->getResultType()->clone();
  }
  return PFTy->clone();
}

typedef std::vector<TypeInfo*> UpRefStack;
static TypeInfo* ResolveUpReference(TypeInfo* Ty, UpRefStack* stack) {
  assert(Ty->isUpReference() && "Can't resolve a non-upreference");
  unsigned upref = atoi(&((Ty->getNewTy().c_str())[1])); // skip the slash
  assert(upref < stack->size() && "Invalid up reference");
  return (*stack)[upref - stack->size() - 1];
}

static TypeInfo* getGEPIndexedType(TypeInfo* PTy, ValueList* idxs) {
  TypeInfo* Result = ResolveType(PTy);
  assert(PTy->isPointer() && "GEP Operand is not a pointer?");
  UpRefStack stack;
  for (unsigned i = 0; i < idxs->size(); ++i) {
    if (Result->isComposite()) {
      Result = Result->getIndexedType((*idxs)[i]);
      ResolveType(Result);
      stack.push_back(Result);
    } else
      yyerror("Invalid type for index");
  }
  // Resolve upreferences so we can return a more natural type
  if (Result->isPointer()) {
    if (Result->getElementType()->isUpReference()) {
      stack.push_back(Result);
      Result = ResolveUpReference(Result->getElementType(), &stack);
    }
  } else if (Result->isUpReference()) {
    Result = ResolveUpReference(Result->getElementType(), &stack);
  }
  return Result->getPointerType();
}

static std::string makeUniqueName(const std::string *Name, bool isSigned) {
  const char *suffix = ".u";
  if (isSigned)
    suffix = ".s";
  if ((*Name)[Name->size()-1] == '"') {
    std::string Result(*Name);
    Result.insert(Name->size()-1, suffix);
    return Result;
  }
  return *Name + suffix;
}

// This function handles appending .u or .s to integer value names that
// were previously unsigned or signed, respectively. This avoids name
// collisions since the unsigned and signed type planes have collapsed
// into a single signless type plane.
static std::string getUniqueName(const std::string *Name, TypeInfo* Ty) {
  // If its not a symbolic name, don't modify it, probably a constant val.
  if ((*Name)[0] != '%' && (*Name)[0] != '"')
    return *Name;
  // If its a numeric reference, just leave it alone.
  if (isdigit((*Name)[1]))
    return *Name;

  // Resolve the type
  ResolveType(Ty);

  // Remove as many levels of pointer nesting that we have.
  if (Ty->isPointer()) {
    // Avoid infinite loops in recursive types
    TypeInfo* Last = 0;
    while (Ty->isPointer() && Last != Ty) {
      Last = Ty;
      Ty = Ty->getElementType();
      ResolveType(Ty);
    }
  }

  // Default the result to the current name
  std::string Result = *Name; 

  // Now deal with the underlying type
  if (Ty->isInteger()) {
    // If its an integer type, make the name unique
    Result = makeUniqueName(Name, Ty->isSigned());
  } else if (Ty->isArray() || Ty->isPacked()) {
    Ty = Ty->getElementType();
    if (Ty->isInteger())
      Result = makeUniqueName(Name, Ty->isSigned());
  } else if (Ty->isStruct()) {
    // Scan the fields and count the signed and unsigned fields
    int isSigned = 0;
    for (unsigned i = 0; i < Ty->getNumStructElements(); ++i) {
      TypeInfo* Tmp = Ty->getElement(i);
      if (Tmp->isInteger())
        if (Tmp->isSigned())
          isSigned++;
        else
          isSigned--;
    }
    if (isSigned != 0)
      Result = makeUniqueName(Name, isSigned > 0);
  }
  return Result;
}


#line 401 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
typedef union {
  std::string*    String;
  TypeInfo*       Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
  TypeList*       TypeVec;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		586
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
   266,   272,   278,   282,   285,   291,   296,   299,   301,   305,
   307,   311,   313,   314,   319,   323,   327,   332,   337,   341,
   344,   347,   350,   353,   356,   359,   362,   365,   368,   371,
   378,   384,   393,   400,   407,   414,   422,   430,   437,   444,
   453,   462,   466,   468,   470,   472,   474,   477,   480,   485,
   488,   490,   492,   494,   499,   502,   507,   514,   521,   528,
   535,   539,   544,   545,   547,   549,   551,   555,   559,   563,
   567,   571,   575,   577,   578,   580,   582,   584,   585,   588,
   592,   594,   596,   600,   602,   603,   612,   614,   616,   620,
   622,   624,   628,   629,   631,   633,   637,   638,   640,   642,
   644,   646,   648,   650,   652,   654,   656,   660,   662,   668,
   670,   672,   674,   676,   679,   682,   684,   687,   690,   691,
   693,   695,   697,   700,   703,   707,   717,   727,   736,   751,
   753,   755,   762,   768,   771,   778,   786,   788,   792,   794,
   795,   798,   800,   806,   812,   818,   825,   832,   835,   840,
   845,   852,   857,   862,   869,   876,   879,   887,   889,   892,
   893,   895,   896,   900,   907,   911,   918,   921,   926,   933
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
     0,   161,   162,     0,   159,   161,   192,   162,   160,     0,
   159,   161,   162,   160,     0,   191,   163,     0,   191,     0,
   192,   152,   191,     0,   192,     0,   192,   152,    40,     0,
    40,     0,     0,   189,   156,   196,   158,     0,   189,   156,
   158,     0,   189,   164,    30,     0,   189,   159,   196,   160,
     0,   189,   161,   196,   162,     0,   189,   161,   162,     0,
   189,    22,     0,   189,    23,     0,   189,   221,     0,   189,
   195,     0,   189,    24,     0,   174,   166,     0,   175,   166,
     0,     4,    25,     0,     4,    26,     0,   177,    21,     0,
   173,   154,   194,    39,   189,   155,     0,   127,   154,   194,
   236,   155,     0,   129,   154,   194,   152,   194,   152,   194,
   155,     0,   167,   154,   194,   152,   194,   155,     0,   168,
   154,   194,   152,   194,   155,     0,   169,   154,   194,   152,
   194,   155,     0,   100,   170,   154,   194,   152,   194,   155,
     0,   101,   171,   154,   194,   152,   194,   155,     0,   172,
   154,   194,   152,   194,   155,     0,   135,   154,   194,   152,
   194,   155,     0,   136,   154,   194,   152,   194,   152,   194,
   155,     0,   137,   154,   194,   152,   194,   152,   194,   155,
     0,   196,   152,   194,     0,   194,     0,    35,     0,    36,
     0,   199,     0,   199,   216,     0,   199,   218,     0,   199,
    63,    62,   202,     0,   199,    31,     0,   201,     0,    50,
     0,    58,     0,   201,   178,    27,   187,     0,   201,   218,
     0,   201,    63,    62,   202,     0,   201,   178,   179,   197,
   194,   185,     0,   201,   178,   200,   197,   189,   185,     0,
   201,   178,    45,   197,   189,   185,     0,   201,   178,    47,
   197,   189,   185,     0,   201,    51,   204,     0,   201,    59,
   151,   205,     0,     0,    30,     0,    56,     0,    55,     0,
    53,   151,   203,     0,    54,   151,    18,     0,    52,   151,
    30,     0,    72,   151,    30,     0,   156,   206,   158,     0,
   206,   152,    30,     0,    30,     0,     0,    28,     0,    30,
     0,   207,     0,     0,   189,   208,     0,   210,   152,   209,
     0,   209,     0,   210,     0,   210,   152,    40,     0,    40,
     0,     0,   180,   187,   207,   154,   211,   155,   184,   181,
     0,    32,     0,   161,     0,   179,   212,   213,     0,    33,
     0,   162,     0,   214,   224,   215,     0,     0,    45,     0,
    47,     0,    34,   217,   212,     0,     0,    64,     0,    17,
     0,    18,     0,    21,     0,    25,     0,    26,     0,    22,
     0,    23,     0,    24,     0,   159,   196,   160,     0,   195,
     0,    62,   219,    30,   152,    30,     0,   165,     0,   207,
     0,   221,     0,   220,     0,   189,   222,     0,   224,   225,
     0,   225,     0,   226,   228,     0,   226,   230,     0,     0,
    29,     0,    78,     0,    77,     0,    73,   223,     0,    73,
     3,     0,    74,    15,   222,     0,    74,     4,   222,   152,
    15,   222,   152,    15,   222,     0,    75,   176,   222,   152,
    15,   222,   156,   229,   158,     0,    75,   176,   222,   152,
    15,   222,   156,   158,     0,   178,    76,   180,   187,   222,
   154,   233,   155,    39,    15,   222,   227,    15,   222,     0,
   227,     0,    79,     0,   229,   176,   220,   152,    15,   222,
     0,   176,   220,   152,    15,   222,     0,   178,   235,     0,
   189,   156,   222,   152,   222,   158,     0,   231,   152,   156,
   222,   152,   222,   158,     0,   223,     0,   232,   152,   223,
     0,   232,     0,     0,    61,    60,     0,    60,     0,   167,
   189,   222,   152,   222,     0,   168,   189,   222,   152,   222,
     0,   169,   189,   222,   152,   222,     0,   100,   170,   189,
   222,   152,   222,     0,   101,   171,   189,   222,   152,   222,
     0,    49,   223,     0,   172,   223,   152,   223,     0,   173,
   223,    39,   189,     0,   129,   223,   152,   223,   152,   223,
     0,   134,   223,   152,   189,     0,   135,   223,   152,   223,
     0,   136,   223,   152,   223,   152,   223,     0,   137,   223,
   152,   223,   152,   223,     0,   128,   231,     0,   234,   180,
   187,   222,   154,   233,   155,     0,   238,     0,   152,   232,
     0,     0,    38,     0,     0,   122,   189,   182,     0,   122,
   189,   152,    10,   222,   182,     0,   123,   189,   182,     0,
   123,   189,   152,    10,   222,   182,     0,   124,   223,     0,
   237,   125,   189,   222,     0,   237,   126,   223,   152,   189,
   222,     0,   127,   189,   222,   236,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   466,   466,   467,   467,   471,   471,   471,   471,   471,   471,
   471,   472,   472,   472,   472,   473,   473,   473,   474,   474,
   474,   474,   474,   474,   475,   475,   475,   475,   475,   475,
   475,   475,   475,   475,   476,   476,   476,   476,   476,   476,
   476,   476,   476,   476,   477,   477,   477,   477,   477,   477,
   478,   478,   478,   478,   479,   479,   479,   479,   479,   479,
   479,   479,   480,   480,   480,   480,   480,   485,   485,   485,
   485,   486,   486,   486,   486,   487,   487,   488,   488,   491,
   494,   499,   499,   499,   499,   499,   499,   500,   501,   504,
   504,   504,   504,   504,   505,   506,   511,   516,   517,   520,
   521,   529,   535,   536,   539,   540,   549,   550,   563,   563,
   564,   564,   565,   569,   569,   569,   569,   569,   569,   569,
   570,   570,   570,   570,   570,   572,   575,   578,   581,   585,
   598,   604,   610,   620,   623,   633,   636,   644,   648,   655,
   656,   661,   666,   676,   682,   687,   693,   699,   705,   710,
   716,   722,   729,   735,   741,   747,   753,   759,   765,   773,
   786,   798,   803,   809,   814,   820,   825,   830,   838,   843,
   848,   858,   863,   868,   868,   878,   883,   886,   891,   895,
   899,   901,   901,   904,   914,   919,   924,   934,   944,   954,
   964,   969,   974,   979,   981,   981,   984,   989,   996,  1001,
  1008,  1015,  1020,  1021,  1029,  1029,  1030,  1030,  1032,  1041,
  1045,  1049,  1052,  1057,  1060,  1063,  1081,  1082,  1085,  1096,
  1097,  1099,  1108,  1109,  1110,  1114,  1127,  1128,  1131,  1131,
  1131,  1131,  1131,  1131,  1131,  1132,  1133,  1138,  1139,  1148,
  1148,  1152,  1157,  1167,  1177,  1180,  1188,  1192,  1197,  1200,
  1206,  1206,  1208,  1213,  1218,  1223,  1232,  1240,  1247,  1270,
  1275,  1281,  1287,  1295,  1313,  1321,  1330,  1334,  1341,  1342,
  1346,  1351,  1354,  1363,  1371,  1380,  1388,  1396,  1401,  1410,
  1438,  1444,  1450,  1457,  1463,  1469,  1475,  1493,  1498,  1499,
  1503,  1504,  1507,  1515,  1524,  1532,  1541,  1547,  1556,  1565
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
   191,   191,   191,   191,   191,   191,   191,   192,   192,   193,
   193,   193,   193,   194,   194,   194,   194,   194,   194,   194,
   194,   194,   194,   194,   194,   194,   194,   194,   194,   195,
   195,   195,   195,   195,   195,   195,   195,   195,   195,   195,
   195,   196,   196,   197,   197,   198,   199,   199,   199,   199,
   199,   200,   200,   201,   201,   201,   201,   201,   201,   201,
   201,   201,   201,   202,   203,   203,   204,   204,   204,   204,
   205,   206,   206,   206,   207,   207,   208,   208,   209,   210,
   210,   211,   211,   211,   211,   212,   213,   213,   214,   215,
   215,   216,   217,   217,   217,   218,   219,   219,   220,   220,
   220,   220,   220,   220,   220,   220,   220,   220,   220,   221,
   221,   222,   222,   223,   224,   224,   225,   226,   226,   226,
   227,   227,   228,   228,   228,   228,   228,   228,   228,   228,
   228,   229,   229,   230,   231,   231,   232,   232,   233,   233,
   234,   234,   235,   235,   235,   235,   235,   235,   235,   235,
   235,   235,   235,   235,   235,   235,   235,   235,   236,   236,
   237,   237,   238,   238,   238,   238,   238,   238,   238,   238
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
     5,     5,     3,     2,     5,     4,     2,     1,     3,     1,
     3,     1,     0,     4,     3,     3,     4,     4,     3,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     6,
     5,     8,     6,     6,     6,     7,     7,     6,     6,     8,
     8,     3,     1,     1,     1,     1,     2,     2,     4,     2,
     1,     1,     1,     4,     2,     4,     6,     6,     6,     6,
     3,     4,     0,     1,     1,     1,     3,     3,     3,     3,
     3,     3,     1,     0,     1,     1,     1,     0,     2,     3,
     1,     1,     3,     1,     0,     8,     1,     1,     3,     1,
     1,     3,     0,     1,     1,     3,     0,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     3,     1,     5,     1,
     1,     1,     1,     2,     2,     1,     2,     2,     0,     1,
     1,     1,     2,     2,     3,     9,     9,     8,    14,     1,
     1,     6,     5,     2,     6,     7,     1,     3,     1,     0,
     2,     1,     5,     5,     5,     6,     6,     2,     4,     4,
     6,     4,     4,     6,     6,     2,     7,     1,     2,     0,
     1,     0,     3,     6,     3,     6,     2,     4,     6,     4
};

static const short yydefact[] = {   193,
    89,   181,   180,   223,    82,    83,    84,    86,    87,    88,
    85,     0,    97,   249,   177,   178,   205,   206,     0,     0,
     0,    89,     0,   185,   224,   225,    97,     0,     0,    90,
    91,    92,    93,    94,    95,     0,     0,   250,   249,   246,
    81,     0,     0,     0,     0,   191,     0,     0,     0,     0,
     0,   182,   183,     0,     0,    80,   226,   194,   179,    96,
   110,   114,   115,   116,   117,   118,   119,   120,   121,   122,
   123,   124,   125,   126,     1,     2,     0,     0,     0,     0,
   240,     0,     0,   109,   128,   113,   241,   127,   217,   218,
   219,   220,   221,   222,   245,     0,     0,     0,   252,   251,
   261,   292,   260,   247,   248,     0,     0,     0,     0,   204,
   192,   186,   184,   174,   175,     0,     0,     0,     0,   129,
     0,     0,     0,   112,   134,   138,     0,     0,   143,   137,
   254,     0,   253,     0,     0,    71,    75,    70,    74,    69,
    73,    68,    72,    76,    77,     0,   291,     0,   272,     0,
    97,     5,     6,     7,     8,     9,    10,    11,    12,    13,
    14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
    24,     0,     0,     0,     0,     0,     0,     0,     0,    51,
    52,    53,    54,     0,     0,     0,     0,    67,    55,    56,
    57,    58,    59,    60,    61,    62,    63,    64,    65,    66,
     0,     0,     0,     0,     0,    97,   264,     0,   288,   199,
   196,   195,   197,   198,   200,   203,     0,   105,   105,   114,
   115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     0,     0,     0,     0,   105,   105,     0,     0,     0,     0,
     0,   133,   215,   142,   140,     0,   229,   230,   231,   234,
   235,   236,   232,   233,   227,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   238,   243,
   242,   244,     0,   255,     0,   278,   271,     0,    25,    26,
    27,    28,    29,    30,    31,    32,    33,    34,     0,    49,
    50,    35,    36,    37,    38,    39,    40,    41,    42,    43,
    44,    45,    46,    47,    48,     0,   100,   100,   297,     0,
     0,   286,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   201,     0,   189,   190,
   157,   158,     3,     4,   155,   156,   159,   150,   151,   154,
     0,     0,     0,     0,   153,   152,   187,   188,   111,   111,
   136,     0,   139,   214,   208,   211,   212,     0,     0,   130,
   228,     0,     0,     0,     0,     0,     0,     0,     0,   173,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   293,     0,   295,   290,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   202,     0,     0,   107,   105,   145,     0,     0,   149,
     0,   146,   131,   132,   135,   207,   209,     0,   103,   141,
     0,     0,     0,   290,     0,     0,     0,     0,     0,   237,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   300,     0,     0,     0,   282,   283,
     0,     0,     0,     0,     0,   279,   280,     0,   298,     0,
   102,   108,   106,   144,   147,   148,   213,   210,   104,    98,
     0,     0,     0,     0,     0,     0,     0,     0,   172,     0,
     0,     0,     0,     0,     0,     0,   270,     0,     0,   100,
   101,   100,   267,   289,     0,     0,     0,     0,     0,   273,
   274,   275,   270,     0,     0,   216,   239,     0,     0,   161,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   269,     0,   276,   277,     0,   294,   296,     0,     0,
     0,   281,   284,   285,     0,   299,    99,     0,     0,     0,
   169,     0,     0,   163,   164,   165,   168,   160,     0,   258,
     0,     0,     0,   268,   265,     0,   287,   166,   167,     0,
     0,     0,   256,     0,   257,     0,     0,   266,   162,   170,
   171,     0,     0,     0,     0,     0,     0,   263,     0,     0,
   262,     0,   259,     0,     0,     0
};

static const short yydefgoto[] = {    81,
   335,   264,   265,   266,   289,   306,   267,   268,   231,   232,
   146,   233,    22,    13,    36,   506,   383,   405,   470,   329,
   406,    82,    83,   234,    85,    86,   127,   246,   370,   269,
   371,   116,   584,     1,    55,     2,    59,   213,    46,   111,
   217,    87,   417,   356,   357,   358,    37,    91,    14,    94,
    15,    27,    16,   362,   270,    88,   272,   493,    39,    40,
    41,   103,   104,   552,   105,   312,   522,   523,   206,   207,
   445,   208,   209
};

static const short yypact[] = {-32768,
   217,   541,-32768,   116,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,    16,   111,    17,-32768,-32768,-32768,-32768,    34,   -90,
    61,    26,    19,-32768,-32768,-32768,   111,   144,   172,-32768,
-32768,-32768,-32768,-32768,-32768,   871,   -26,-32768,   -18,-32768,
    47,    81,    87,    93,    98,-32768,    96,   144,   871,    69,
    69,-32768,-32768,    69,    69,-32768,-32768,-32768,-32768,-32768,
   100,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,   237,   238,    -9,   215,
-32768,   134,   103,-32768,-32768,  -111,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,   909,    36,   148,-32768,-32768,
-32768,  1343,-32768,-32768,-32768,   228,    40,   248,   239,   241,
-32768,-32768,-32768,-32768,-32768,   937,   937,   965,   937,-32768,
   110,   115,   506,-32768,-32768,  -111,   -98,   114,   673,-32768,
   100,  1141,-32768,  1141,  1141,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,  1141,-32768,   937,-32768,   213,
   111,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,   195,    84,   937,   937,   937,   937,   937,   937,-32768,
-32768,-32768,-32768,   937,   937,   937,   937,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
   937,   937,   937,   937,   937,   111,-32768,     6,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,   -93,   122,   122,    91,
   119,   129,   166,   169,   191,   193,   219,   223,   258,   260,
   229,   229,   261,  1052,   122,   122,   937,   937,   124,   -73,
   937,-32768,   711,-32768,   133,   131,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   226,   195,    84,   137,   138,   140,
   150,   154,   965,   156,   158,   167,   168,   170,-32768,-32768,
-32768,-32768,   171,-32768,   173,-32768,-32768,   871,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   937,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,   937,   174,   175,-32768,  1141,
   164,   176,   177,   178,   179,   182,   183,  1141,  1141,  1141,
   184,   299,   871,   937,   937,   310,-32768,    -8,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
   749,   965,   614,   311,-32768,-32768,-32768,-32768,   -55,  -115,
-32768,   185,  -111,-32768,   134,-32768,   190,   189,   777,-32768,
-32768,   316,   194,   198,   965,   965,   965,   965,   965,-32768,
  -102,   965,   965,   965,   965,   965,   332,   344,  1141,  1141,
  1141,    -2,-32768,     9,-32768,   208,  1141,   207,   937,   937,
   937,   937,   937,   212,   214,   218,   937,   937,  1141,  1141,
   220,-32768,   335,   349,-32768,   122,-32768,   -61,   -58,-32768,
   -70,-32768,-32768,-32768,-32768,-32768,-32768,   833,   336,-32768,
   231,   965,   965,   208,   235,   236,   240,   242,   965,-32768,
   243,   244,   245,   249,   330,  1141,  1141,   221,   250,   251,
  1141,   363,  1141,   937,-32768,   252,  1141,   253,-32768,-32768,
   254,   255,  1141,  1141,  1141,-32768,-32768,   256,-32768,   937,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   333,
   359,   257,   262,   263,   965,   965,   965,   965,-32768,   965,
   965,   965,   965,   937,   264,   259,   937,  1141,  1141,   265,
-32768,   265,-32768,   267,  1141,   268,   937,   937,   937,-32768,
-32768,-32768,   937,  1141,   375,-32768,-32768,   965,   965,-32768,
   269,   273,   270,   271,   274,   275,   276,   277,   278,   419,
    15,   267,   282,-32768,-32768,   381,-32768,-32768,   937,   281,
  1141,-32768,-32768,-32768,   286,-32768,-32768,   289,   292,   965,
-32768,   965,   965,-32768,-32768,-32768,-32768,-32768,  1141,-32768,
  1230,    27,   410,-32768,-32768,   297,-32768,-32768,-32768,   295,
   303,   304,-32768,   308,-32768,  1230,   449,-32768,-32768,-32768,
-32768,   450,   314,  1141,  1141,   453,   135,-32768,  1141,   454,
-32768,  1141,-32768,   472,   473,-32768
};

static const short yypgoto[] = {-32768,
   247,   372,   374,   378,   225,   227,   380,   383,   -96,   -95,
  -505,-32768,   436,   461,  -139,-32768,  -301,    67,-32768,  -218,
-32768,   -45,-32768,   -36,-32768,   -66,   -16,-32768,   132,   290,
  -214,    60,-32768,-32768,-32768,-32768,   441,-32768,-32768,-32768,
-32768,     3,-32768,    72,-32768,-32768,   464,-32768,-32768,-32768,
-32768,-32768,   490,-32768,  -510,  -104,    -1,   -86,-32768,   456,
-32768,   -81,-32768,-32768,-32768,-32768,    58,    20,-32768,-32768,
   104,-32768,-32768
};


#define	YYLAST		1493


static const short yytable[] = {    84,
   330,   144,   145,   113,    23,    89,   385,   441,   122,   133,
    38,   278,    84,   126,    92,   551,   347,   348,   443,   136,
   137,   138,   139,   140,   141,   142,   143,   271,   403,   271,
   271,   136,   137,   138,   139,   140,   141,   142,   143,   134,
   564,   271,  -111,    23,   414,    38,   566,   130,   404,   429,
   135,   130,    49,   241,   442,   573,   126,   430,   326,   132,
    47,   276,   126,   242,   327,   442,   323,     5,     6,     7,
    50,     9,    51,    11,    17,    52,    18,    28,   241,   218,
   219,   429,   236,    53,   128,    42,    43,    44,   352,   309,
   429,   466,   313,   429,   211,   212,   464,   314,   315,   316,
   317,   465,   413,   114,   115,    45,   240,   130,   290,   291,
   117,   132,   245,   118,   119,   331,   332,   321,   322,    96,
    97,    98,    48,    99,   100,   101,   408,   409,   411,   346,
   324,   325,   273,   274,    90,   -71,   -71,   307,   308,   132,
   310,   311,   132,    93,   275,   -75,   -75,   132,   132,   132,
   132,   123,   136,   137,   138,   139,   140,   141,   142,   143,
    25,    17,    26,    18,   318,   319,   320,   132,   132,    56,
   349,   350,   550,    58,   353,    29,    30,    31,    32,    33,
    34,    35,   -70,   -70,   565,   -74,   -74,   463,   527,    60,
   528,   292,   293,   294,   295,   296,   297,   298,   299,   300,
   301,   302,   303,   304,   305,   271,   355,   -69,   -69,   -73,
   -73,    99,   100,   271,   271,   271,  -176,   124,    62,    63,
    64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
    74,   106,   379,    75,    76,   -68,   -68,   107,   401,   -72,
   -72,    84,    17,   108,    18,   333,   334,     3,   109,   235,
     4,   110,   380,  -112,   120,   121,   129,   210,     5,     6,
     7,     8,     9,    10,    11,   214,   237,   243,   215,   381,
   216,   238,   277,   328,   271,   271,   271,   399,   -78,    12,
   -79,   337,   271,   351,   359,   360,    84,   400,   132,   361,
   365,   366,   353,   367,   271,   271,   279,   280,   281,   282,
   283,   284,   448,   368,   450,   451,   452,   369,   386,   372,
   456,   373,   285,   286,   287,   288,   394,   395,   396,   387,
   374,   375,   377,   376,   378,   382,   384,   388,   389,   390,
   391,   271,   271,   392,   393,   397,   271,   398,   271,   402,
   412,   418,   271,   419,   415,   421,   436,   422,   271,   271,
   271,   423,   132,   449,   132,   132,   132,   416,   437,   444,
   132,   457,   447,   453,   461,   454,   462,    77,   484,   455,
    78,   460,   403,    79,   487,    80,   125,   438,   439,   440,
   491,   355,   471,   271,   271,   446,   475,   476,   507,   505,
   271,   477,   537,   478,   480,   481,   482,   458,   459,   271,
   483,   488,   489,   495,   497,   498,   499,   132,   508,   503,
   532,   533,   534,   509,   521,   520,   526,   510,   529,   531,
   540,   542,   543,   504,   144,   145,   271,   541,   544,   545,
   546,   547,   548,   549,   485,   486,   553,   442,   555,   490,
   557,   492,   554,   558,   271,   496,   559,   519,   567,   569,
   132,   500,   501,   502,   568,   144,   145,   570,   571,   572,
   132,   132,   132,   574,   575,   576,   132,   579,   582,   271,
   271,   585,   586,   201,   271,   202,   102,   271,   336,   203,
   363,   204,    54,   364,   205,   469,   524,   525,   112,   468,
    57,    24,   132,   530,    95,   580,   424,   425,   426,   427,
   428,   494,   536,   431,   432,   433,   434,   435,   124,    62,
    63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
    73,    74,   535,   345,    75,    76,     0,   474,     0,   556,
     0,     0,     0,    17,     0,    18,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   563,     0,     0,
     0,     0,     0,   472,   473,     0,     0,     0,     0,     0,
   479,     0,     0,     0,     0,     0,     0,   -81,    17,     0,
    18,     0,   577,   578,     4,   -81,   -81,   581,     0,     0,
   583,     0,   -81,   -81,   -81,   -81,   -81,   -81,   -81,     0,
   -81,    19,     0,     0,     0,     0,     0,     0,   -81,    20,
     0,     0,     0,    21,     0,     0,   511,   512,   513,   514,
     0,   515,   516,   517,   518,     0,   124,   220,   221,   222,
   223,   224,   225,   226,   227,   228,   229,   230,    73,    74,
     0,     0,    75,    76,     0,     0,     0,     0,     0,   538,
   539,    17,     0,    18,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    77,     0,
     0,    78,     0,     0,    79,     0,    80,   239,     0,     0,
     0,   560,     0,   561,   562,   124,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,     0,
     0,    75,    76,     0,     0,     0,     0,     0,     0,     0,
    17,     0,    18,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   244,   124,    62,    63,    64,    65,    66,    67,
    68,    69,    70,    71,    72,    73,    74,     0,     0,    75,
    76,     0,     0,     0,     0,     0,     0,     0,    17,     0,
    18,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   354,   124,   220,   221,   222,   223,   224,   225,   226,   227,
   228,   229,   230,    73,    74,     0,    77,    75,    76,    78,
     0,     0,    79,     0,    80,   410,    17,     0,    18,   124,
    62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
    72,    73,    74,     0,     0,    75,    76,     0,     0,     0,
     0,     0,     0,     0,    17,     0,    18,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   420,     0,     0,     0,
     0,     0,     0,     0,     0,    77,     0,     0,    78,     0,
     0,    79,     0,    80,     0,   124,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,     0,
     0,    75,    76,     0,     0,     0,     0,     0,     0,     0,
    17,     0,    18,    77,     0,     0,    78,     0,     0,    79,
     0,    80,   467,    61,    62,    63,    64,    65,    66,    67,
    68,    69,    70,    71,    72,    73,    74,     0,     0,    75,
    76,     0,     0,     0,     0,     0,     0,     0,    17,     0,
    18,    77,     0,     0,    78,     0,   407,    79,     0,    80,
     0,   131,    62,    63,    64,    65,    66,    67,    68,    69,
    70,    71,    72,    73,    74,     0,     0,    75,    76,    77,
     0,     0,    78,     0,     0,    79,    17,    80,    18,   124,
    62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
    72,    73,    74,     0,     0,    75,    76,     0,     0,     0,
     0,     0,     0,     0,    17,     0,    18,   124,   220,   221,
   222,   223,   224,   225,   226,   227,   228,   229,   230,    73,
    74,     0,     0,    75,    76,    77,     0,     0,    78,     0,
     0,    79,    17,    80,    18,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,    77,     0,     0,    78,     0,     0,    79,
     0,    80,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    77,     0,     0,    78,     0,     0,    79,     0,    80,
    75,    76,     0,   338,   339,   340,     0,     0,     0,    17,
     0,    18,     0,     0,     0,     0,     0,     0,     0,    77,
     0,     0,    78,     0,     0,    79,     0,    80,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    77,     0,     0,
    78,     0,     0,    79,     0,    80,     0,     0,     0,     0,
     0,   152,   153,   154,   155,   156,   157,   158,   159,   160,
   161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
   171,   256,   257,     0,     0,     0,     0,   247,   248,    75,
    76,   249,   250,   251,   252,   253,   254,     0,    17,     0,
    18,     0,     0,     0,     0,     0,     0,     0,   258,     0,
   259,   180,   181,   182,   183,     0,   260,   261,   262,   188,
   189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
   199,   200,   255,     0,     0,     0,     0,   341,     0,     0,
   342,     0,   343,     0,     0,   344,     0,     0,     0,     0,
   152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
   162,   163,   164,   165,   166,   167,   168,   169,   170,   171,
   256,   257,     0,     0,     0,     0,   247,   248,     0,     0,
   249,   250,   251,   252,   253,   254,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   258,     0,   259,
   180,   181,   182,   183,     0,   260,   261,   262,   188,   189,
   190,   191,   192,   193,   194,   195,   196,   197,   198,   199,
   200,   255,     0,     0,     0,     0,     0,     0,     0,   263,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   152,
   153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
   163,   164,   165,   166,   167,   168,   169,   170,   171,   256,
   257,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   258,     0,   259,   180,
   181,   182,   183,     0,   260,   261,   262,   188,   189,   190,
   191,   192,   193,   194,   195,   196,   197,   198,   199,   200,
   147,     0,     0,     0,     0,     0,     0,     0,   263,     0,
     0,   148,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   149,   150,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   151,     0,
     0,     0,   152,   153,   154,   155,   156,   157,   158,   159,
   160,   161,   162,   163,   164,   165,   166,   167,   168,   169,
   170,   171,   172,   173,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   174,   175,   176,     0,     0,   177,
   178,   179,   180,   181,   182,   183,   184,   185,   186,   187,
   188,   189,   190,   191,   192,   193,   194,   195,   196,   197,
   198,   199,   200
};

static const short yycheck[] = {    36,
   219,    98,    98,    49,     2,    32,   308,    10,    18,    96,
    29,   151,    49,    80,    33,   521,   235,   236,    10,     5,
     6,     7,     8,     9,    10,    11,    12,   132,    37,   134,
   135,     5,     6,     7,     8,     9,    10,    11,    12,     4,
   551,   146,   154,    41,   160,    29,   552,   163,    57,   152,
    15,   163,    27,   152,    57,   566,   123,   160,   152,    96,
   151,   148,   129,   162,   158,    57,   206,    42,    43,    44,
    45,    46,    47,    48,    28,    50,    30,    62,   152,   116,
   117,   152,   119,    58,    82,    52,    53,    54,   162,   176,
   152,   162,   179,   152,    55,    56,   158,   184,   185,   186,
   187,   160,   158,    35,    36,    72,   123,   163,    25,    26,
    51,   148,   129,    54,    55,    25,    26,   204,   205,    73,
    74,    75,    62,    77,    78,    79,   341,   342,   343,   234,
   125,   126,   134,   135,   161,    17,    18,   174,   175,   176,
   177,   178,   179,   162,   146,    17,    18,   184,   185,   186,
   187,   161,     5,     6,     7,     8,     9,    10,    11,    12,
    45,    28,    47,    30,   201,   202,   203,   204,   205,   151,
   237,   238,   158,    30,   241,    65,    66,    67,    68,    69,
    70,    71,    17,    18,   158,    17,    18,   406,   490,    18,
   492,   108,   109,   110,   111,   112,   113,   114,   115,   116,
   117,   118,   119,   120,   121,   310,   243,    17,    18,    17,
    18,    77,    78,   318,   319,   320,     0,     3,     4,     5,
     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,   151,   278,    19,    20,    17,    18,   151,   325,    17,
    18,   278,    28,   151,    30,    17,    18,    31,   151,   118,
    34,   156,   289,   154,    18,    18,   154,    30,    42,    43,
    44,    45,    46,    47,    48,    18,   157,   154,    30,   306,
    30,   157,    60,   152,   379,   380,   381,   323,    21,    63,
    21,    21,   387,   160,   152,   155,   323,   324,   325,    64,
   154,   154,   359,   154,   399,   400,   102,   103,   104,   105,
   106,   107,   389,   154,   391,   392,   393,   154,   310,   154,
   397,   154,   118,   119,   120,   121,   318,   319,   320,   156,
   154,   154,   152,   154,   152,   152,   152,   152,   152,   152,
   152,   436,   437,   152,   152,   152,   441,    39,   443,    30,
    30,   152,   447,   155,   160,    30,    15,   154,   453,   454,
   455,   154,   389,   390,   391,   392,   393,   355,    15,   152,
   397,   398,   156,   152,    30,   152,    18,   153,    39,   152,
   156,   152,    37,   159,   154,   161,   162,   379,   380,   381,
    18,   418,   152,   488,   489,   387,   152,   152,    30,    57,
   495,   152,    18,   152,   152,   152,   152,   399,   400,   504,
   152,   152,   152,   152,   152,   152,   152,   444,   152,   154,
   497,   498,   499,   152,   156,   152,   152,   155,   152,   152,
   152,   152,   152,   460,   521,   521,   531,   155,   155,   155,
   155,   155,   155,    15,   436,   437,   155,    57,   158,   441,
   155,   443,   529,   155,   549,   447,   155,   484,    39,   155,
   487,   453,   454,   455,   158,   552,   552,   155,   155,   152,
   497,   498,   499,    15,    15,   152,   503,    15,    15,   574,
   575,     0,     0,   102,   579,   102,    41,   582,   232,   102,
   256,   102,    22,   257,   102,   419,   488,   489,    48,   418,
    27,     2,   529,   495,    39,   577,   365,   366,   367,   368,
   369,   444,   504,   372,   373,   374,   375,   376,     3,     4,
     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
    15,    16,   503,   234,    19,    20,    -1,   424,    -1,   531,
    -1,    -1,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   549,    -1,    -1,
    -1,    -1,    -1,   422,   423,    -1,    -1,    -1,    -1,    -1,
   429,    -1,    -1,    -1,    -1,    -1,    -1,    27,    28,    -1,
    30,    -1,   574,   575,    34,    35,    36,   579,    -1,    -1,
   582,    -1,    42,    43,    44,    45,    46,    47,    48,    -1,
    50,    51,    -1,    -1,    -1,    -1,    -1,    -1,    58,    59,
    -1,    -1,    -1,    63,    -1,    -1,   475,   476,   477,   478,
    -1,   480,   481,   482,   483,    -1,     3,     4,     5,     6,
     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,   508,
   509,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,
    -1,   156,    -1,    -1,   159,    -1,   161,   162,    -1,    -1,
    -1,   540,    -1,   542,   543,     3,     4,     5,     6,     7,
     8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    40,     3,     4,     5,     6,     7,     8,     9,
    10,    11,    12,    13,    14,    15,    16,    -1,    -1,    19,
    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,
    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    40,     3,     4,     5,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    -1,   153,    19,    20,   156,
    -1,    -1,   159,    -1,   161,   162,    28,    -1,    30,     3,
     4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,   156,    -1,
    -1,   159,    -1,   161,    -1,     3,     4,     5,     6,     7,
     8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    28,    -1,    30,   153,    -1,    -1,   156,    -1,    -1,   159,
    -1,   161,    40,     3,     4,     5,     6,     7,     8,     9,
    10,    11,    12,    13,    14,    15,    16,    -1,    -1,    19,
    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,
    30,   153,    -1,    -1,   156,    -1,   158,   159,    -1,   161,
    -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
    12,    13,    14,    15,    16,    -1,    -1,    19,    20,   153,
    -1,    -1,   156,    -1,    -1,   159,    28,   161,    30,     3,
     4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    28,    -1,    30,     3,     4,     5,
     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,    -1,    -1,    19,    20,   153,    -1,    -1,   156,    -1,
    -1,   159,    28,   161,    30,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,   153,    -1,    -1,   156,    -1,    -1,   159,
    -1,   161,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   153,    -1,    -1,   156,    -1,    -1,   159,    -1,   161,
    19,    20,    -1,    22,    23,    24,    -1,    -1,    -1,    28,
    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   153,
    -1,    -1,   156,    -1,    -1,   159,    -1,   161,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,
   156,    -1,    -1,   159,    -1,   161,    -1,    -1,    -1,    -1,
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
#line 491 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 81:
#line 494 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = new std::string(""); 
  ;
    break;}
case 89:
#line 501 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(""); ;
    break;}
case 96:
#line 506 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    *yyvsp[-1].String += *yyvsp[0].String; 
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String; 
    ;
    break;}
case 97:
#line 511 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(""); ;
    break;}
case 98:
#line 516 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 99:
#line 517 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ *yyvsp[-1].String += " " + *yyvsp[0].String; delete yyvsp[0].String; yyval.String = yyvsp[-1].String; ;
    break;}
case 100:
#line 520 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 101:
#line 521 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyvsp[-1].String->insert(0, ", "); 
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 102:
#line 529 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 103:
#line 535 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 105:
#line 539 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 106:
#line 540 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
      yyvsp[-1].String->insert(0, ", ");
      if (!yyvsp[0].String->empty())
        *yyvsp[-1].String += " " + *yyvsp[0].String;
      delete yyvsp[0].String;
      yyval.String = yyvsp[-1].String;
    ;
    break;}
case 108:
#line 550 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
      *yyvsp[-1].String += " " + *yyvsp[0].String;
      delete yyvsp[0].String;
      yyval.String = yyvsp[-1].String;
    ;
    break;}
case 126:
#line 572 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.Type = new TypeInfo(yyvsp[0].String, OpaqueTy);
  ;
    break;}
case 127:
#line 575 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.Type = new TypeInfo(yyvsp[0].String, UnresolvedTy);
  ;
    break;}
case 128:
#line 578 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.Type = yyvsp[0].Type; 
  ;
    break;}
case 129:
#line 581 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                   // Type UpReference
    yyvsp[0].String->insert(0, "\\");
    yyval.Type = new TypeInfo(yyvsp[0].String, UpRefTy);
  ;
    break;}
case 130:
#line 585 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{           // Function derived type?
    std::string newTy( yyvsp[-3].Type->getNewTy() + "(");
    for (unsigned i = 0; i < yyvsp[-1].TypeVec->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      if ((*yyvsp[-1].TypeVec)[i]->isVoid())
        newTy += "...";
      else
        newTy += (*yyvsp[-1].TypeVec)[i]->getNewTy();
    }
    newTy += ")";
    yyval.Type = new TypeInfo(new std::string(newTy), yyvsp[-3].Type, yyvsp[-1].TypeVec);
  ;
    break;}
case 131:
#line 598 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{          // Sized array type?
    yyvsp[-3].String->insert(0,"[ ");
    *yyvsp[-3].String += " x " + yyvsp[-1].Type->getNewTy() + " ]";
    uint64_t elems = atoi(yyvsp[-3].String->c_str());
    yyval.Type = new TypeInfo(yyvsp[-3].String, ArrayTy, yyvsp[-1].Type, elems);
  ;
    break;}
case 132:
#line 604 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{          // Packed array type?
    yyvsp[-3].String->insert(0,"< ");
    *yyvsp[-3].String += " x " + yyvsp[-1].Type->getNewTy() + " >";
    uint64_t elems = atoi(yyvsp[-3].String->c_str());
    yyval.Type = new TypeInfo(yyvsp[-3].String, PackedTy, yyvsp[-1].Type, elems);
  ;
    break;}
case 133:
#line 610 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                        // Structure type?
    std::string newTy("{");
    for (unsigned i = 0; i < yyvsp[-1].TypeVec->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      newTy += (*yyvsp[-1].TypeVec)[i]->getNewTy();
    }
    newTy += "}";
    yyval.Type = new TypeInfo(new std::string(newTy), StructTy, yyvsp[-1].TypeVec);
  ;
    break;}
case 134:
#line 620 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                                  // Empty structure type?
    yyval.Type = new TypeInfo(new std::string("{}"), StructTy, new TypeList());
  ;
    break;}
case 135:
#line 623 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                // Packed Structure type?
    std::string newTy("<{");
    for (unsigned i = 0; i < yyvsp[-2].TypeVec->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      newTy += (*yyvsp[-2].TypeVec)[i]->getNewTy();
    }
    newTy += "}>";
    yyval.Type = new TypeInfo(new std::string(newTy), PackedStructTy, yyvsp[-2].TypeVec);
  ;
    break;}
case 136:
#line 633 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                          // Empty packed structure type?
    yyval.Type = new TypeInfo(new std::string("<{}>"), PackedStructTy, new TypeList());
  ;
    break;}
case 137:
#line 636 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                             // Pointer type?
    yyval.Type = yyvsp[-1].Type->getPointerType();
  ;
    break;}
case 138:
#line 644 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.TypeVec = new TypeList();
    yyval.TypeVec->push_back(yyvsp[0].Type);
  ;
    break;}
case 139:
#line 648 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.TypeVec = yyvsp[-2].TypeVec;
    yyval.TypeVec->push_back(yyvsp[0].Type);
  ;
    break;}
case 141:
#line 656 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.TypeVec = yyvsp[-2].TypeVec;
    yyval.TypeVec->push_back(new TypeInfo("void",VoidTy));
    delete yyvsp[0].String;
  ;
    break;}
case 142:
#line 661 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.TypeVec = new TypeList();
    yyval.TypeVec->push_back(new TypeInfo("void",VoidTy));
    delete yyvsp[0].String;
  ;
    break;}
case 143:
#line 666 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.TypeVec = new TypeList();
  ;
    break;}
case 144:
#line 676 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ // Nonempty unsized arr
    yyval.Const.type = yyvsp[-3].Type;
    yyval.Const.cnst = new std::string(yyvsp[-3].Type->getNewTy());
    *yyval.Const.cnst += " [ " + *yyvsp[-1].String + " ]";
    delete yyvsp[-1].String;
  ;
    break;}
case 145:
#line 682 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-2].Type;
    yyval.Const.cnst = new std::string(yyvsp[-2].Type->getNewTy());
    *yyval.Const.cnst += "[ ]";
  ;
    break;}
case 146:
#line 687 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-2].Type;
    yyval.Const.cnst = new std::string(yyvsp[-2].Type->getNewTy());
    *yyval.Const.cnst += " c" + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 147:
#line 693 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ // Nonempty unsized arr
    yyval.Const.type = yyvsp[-3].Type;
    yyval.Const.cnst = new std::string(yyvsp[-3].Type->getNewTy());
    *yyval.Const.cnst += " < " + *yyvsp[-1].String + " >";
    delete yyvsp[-1].String;
  ;
    break;}
case 148:
#line 699 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-3].Type;
    yyval.Const.cnst = new std::string(yyvsp[-3].Type->getNewTy());
    *yyval.Const.cnst += " { " + *yyvsp[-1].String + " }";
    delete yyvsp[-1].String;
  ;
    break;}
case 149:
#line 705 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-2].Type;
    yyval.Const.cnst = new std::string(yyvsp[-2].Type->getNewTy());
    *yyval.Const.cnst += " {}";
  ;
    break;}
case 150:
#line 710 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst +=  " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 151:
#line 716 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 152:
#line 722 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[0].String,yyvsp[-1].Type);
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + Name;
    delete yyvsp[0].String;
  ;
    break;}
case 153:
#line 729 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 154:
#line 735 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 155:
#line 741 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{      // integral constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 156:
#line 747 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{            // integral constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 157:
#line 753 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                      // Boolean constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 158:
#line 759 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                     // Boolean constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 159:
#line 765 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                   // Float & Double constants
    yyval.Const.type = yyvsp[-1].Type;
    yyval.Const.cnst = new std::string(yyvsp[-1].Type->getNewTy());
    *yyval.Const.cnst += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 160:
#line 773 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string source = *yyvsp[-3].Const.cnst;
    TypeInfo* DstTy = ResolveType(yyvsp[-1].Type);
    if (*yyvsp[-5].String == "cast") {
      // Call getCastUpgrade to upgrade the old cast
      yyval.String = new std::string(getCastUpgrade(source, yyvsp[-3].Const.type, DstTy, true));
    } else {
      // Nothing to upgrade, just create the cast constant expr
      yyval.String = new std::string(*yyvsp[-5].String);
      *yyval.String += "( " + source + " to " + yyvsp[-1].Type->getNewTy() + ")";
    }
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); delete yyvsp[-2].String;
  ;
    break;}
case 161:
#line 786 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
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
case 162:
#line 798 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-7].String += "(" + *yyvsp[-5].Const.cnst + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-5].Const.destroy(); yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 163:
#line 803 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* op = getDivRemOpcode(*yyvsp[-5].String, yyvsp[-3].Const.type); 
    yyval.String = new std::string(op);
    *yyval.String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
  ;
    break;}
case 164:
#line 809 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 165:
#line 814 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String = getCompareOp(*yyvsp[-5].String, yyvsp[-3].Const.type);
    *yyvsp[-5].String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 166:
#line 820 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += "(" + *yyvsp[-5].String + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 167:
#line 825 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-6].String += "(" + *yyvsp[-5].String + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-6].String;
  ;
    break;}
case 168:
#line 830 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* shiftop = yyvsp[-5].String->c_str();
    if (*yyvsp[-5].String == "shr")
      shiftop = (yyvsp[-3].Const.type->isUnsigned()) ? "lshr" : "ashr";
    yyval.String = new std::string(shiftop);
    *yyval.String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    delete yyvsp[-5].String; yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
  ;
    break;}
case 169:
#line 838 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += "(" + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 170:
#line 843 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-7].String += "(" + *yyvsp[-5].Const.cnst + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-5].Const.destroy(); yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 171:
#line 848 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-7].String += "(" + *yyvsp[-5].Const.cnst + "," + *yyvsp[-3].Const.cnst + "," + *yyvsp[-1].Const.cnst + ")";
    yyvsp[-5].Const.destroy(); yyvsp[-3].Const.destroy(); yyvsp[-1].Const.destroy();
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 172:
#line 858 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].Const.cnst;
    yyvsp[0].Const.destroy();
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 173:
#line 863 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(*yyvsp[0].Const.cnst); yyvsp[0].Const.destroy(); ;
    break;}
case 176:
#line 878 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
;
    break;}
case 177:
#line 883 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 178:
#line 886 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << *yyvsp[0].String << '\n';
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 179:
#line 891 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "module asm " << ' ' << *yyvsp[0].String << '\n';
    yyval.String = 0;
  ;
    break;}
case 180:
#line 895 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "implementation\n";
    yyval.String = 0;
  ;
    break;}
case 181:
#line 899 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = 0; ;
    break;}
case 183:
#line 901 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = yyvsp[0].String; *yyval.String = "external"; ;
    break;}
case 184:
#line 904 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    EnumeratedTypes.push_back(*yyvsp[0].Type);
    if (!yyvsp[-2].String->empty()) {
      NamedTypes[*yyvsp[-2].String] = *yyvsp[0].Type;
      *O << *yyvsp[-2].String << " = ";
    }
    *O << "type " << yyvsp[0].Type->getNewTy() << '\n';
    delete yyvsp[-2].String; delete yyvsp[-1].String;
    yyval.String = 0;
  ;
    break;}
case 185:
#line 914 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{       // Function prototypes can be in const pool
    *O << *yyvsp[0].String << '\n';
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 186:
#line 919 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{  // Asm blocks can be in the const pool
    *O << *yyvsp[-2].String << ' ' << *yyvsp[-1].String << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-2].String; delete yyvsp[-1].String; delete yyvsp[0].String; 
    yyval.String = 0;
  ;
    break;}
case 187:
#line 924 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      std::string Name = getUniqueName(yyvsp[-4].String,yyvsp[-1].Const.type);
      *O << Name << " = ";
      Globals[Name] = *yyvsp[-1].Const.type;
    }
    *O << *yyvsp[-3].String << ' ' << *yyvsp[-2].String << ' ' << *yyvsp[-1].Const.cnst << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; delete yyvsp[0].String; 
    yyval.String = 0;
  ;
    break;}
case 188:
#line 934 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      std::string Name = getUniqueName(yyvsp[-4].String,yyvsp[-1].Type);
      *O << Name << " = ";
      Globals[Name] = *yyvsp[-1].Type;
    }
    *O <<  *yyvsp[-3].String << ' ' << *yyvsp[-2].String << ' ' << yyvsp[-1].Type->getNewTy() << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 189:
#line 944 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      std::string Name = getUniqueName(yyvsp[-4].String,yyvsp[-1].Type);
      *O << Name << " = ";
      Globals[Name] = *yyvsp[-1].Type;
    }
    *O << *yyvsp[-3].String << ' ' << *yyvsp[-2].String << ' ' << yyvsp[-1].Type->getNewTy() << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 190:
#line 954 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-4].String->empty()) {
      std::string Name = getUniqueName(yyvsp[-4].String,yyvsp[-1].Type);
      *O << Name << " = ";
      Globals[Name] = *yyvsp[-1].Type;
    }
    *O << *yyvsp[-3].String << ' ' << *yyvsp[-2].String << ' ' << yyvsp[-1].Type->getNewTy() << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-4].String; delete yyvsp[-3].String; delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 191:
#line 964 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    *O << *yyvsp[-1].String << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-1].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 192:
#line 969 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << *yyvsp[-2].String << " = " << *yyvsp[0].String << '\n';
    delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 193:
#line 974 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyval.String = 0;
  ;
    break;}
case 197:
#line 984 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 198:
#line 989 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    if (*yyvsp[0].String == "64")
      SizeOfPointer = 64;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 199:
#line 996 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 200:
#line 1001 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " = " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 201:
#line 1008 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyvsp[-1].String->insert(0, "[ ");
    *yyvsp[-1].String += " ]";
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 202:
#line 1015 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 204:
#line 1021 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = new std::string();
  ;
    break;}
case 208:
#line 1030 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 209:
#line 1032 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
  yyval.String = new std::string(yyvsp[-1].Type->getNewTy());
  if (!yyvsp[0].String->empty()) {
    std::string Name = getUniqueName(yyvsp[0].String, yyvsp[-1].Type);
    *yyval.String += " " + Name;
  }
  delete yyvsp[0].String;
;
    break;}
case 210:
#line 1041 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", " + *yyvsp[0].String;
    delete yyvsp[0].String;
  ;
    break;}
case 211:
#line 1045 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 212:
#line 1049 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 213:
#line 1052 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += ", ...";
    yyval.String = yyvsp[-2].String;
    delete yyvsp[0].String;
  ;
    break;}
case 214:
#line 1057 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = yyvsp[0].String;
  ;
    break;}
case 215:
#line 1060 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 216:
#line 1063 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-7].String->empty()) {
      *yyvsp[-7].String += " ";
    }
    *yyvsp[-7].String += yyvsp[-6].Type->getNewTy() + " " + *yyvsp[-5].String + "(" + *yyvsp[-3].String + ")";
    if (!yyvsp[-1].String->empty()) {
      *yyvsp[-7].String += " " + *yyvsp[-1].String;
    }
    if (!yyvsp[0].String->empty()) {
      *yyvsp[-7].String += " " + *yyvsp[0].String;
    }
    delete yyvsp[-5].String;
    delete yyvsp[-3].String;
    delete yyvsp[-1].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-7].String;
  ;
    break;}
case 217:
#line 1081 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string("{"); delete yyvsp[0].String; ;
    break;}
case 218:
#line 1082 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string ("{"); ;
    break;}
case 219:
#line 1085 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "define ";
    if (!yyvsp[-2].String->empty()) {
      *O << *yyvsp[-2].String << ' ';
    }
    *O << *yyvsp[-1].String << ' ' << *yyvsp[0].String << '\n';
    delete yyvsp[-2].String; delete yyvsp[-1].String; delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 220:
#line 1096 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string("}"); delete yyvsp[0].String; ;
    break;}
case 221:
#line 1097 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string("}"); ;
    break;}
case 222:
#line 1099 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
  if (yyvsp[-1].String)
    *O << *yyvsp[-1].String;
  *O << *yyvsp[0].String << "\n\n";
  delete yyvsp[-2].String; delete yyvsp[-1].String; delete yyvsp[0].String;
  yyval.String = 0;
;
    break;}
case 223:
#line 1108 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 226:
#line 1114 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    if (!yyvsp[-1].String->empty())
      *yyvsp[-2].String += " " + *yyvsp[-1].String;
    *yyvsp[-2].String += " " + *yyvsp[0].String;
    delete yyvsp[-1].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-2].String;
  ;
    break;}
case 227:
#line 1127 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 237:
#line 1133 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ 
    yyvsp[-1].String->insert(0, "<");
    *yyvsp[-1].String += ">";
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 239:
#line 1139 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-3].String->empty()) {
      *yyvsp[-4].String += " " + *yyvsp[-3].String;
    }
    *yyvsp[-4].String += " " + *yyvsp[-2].String + ", " + *yyvsp[0].String;
    delete yyvsp[-3].String; delete yyvsp[-2].String; delete yyvsp[0].String;
    yyval.String = yyvsp[-4].String;
  ;
    break;}
case 242:
#line 1152 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Value.val = yyvsp[0].String;
    yyval.Value.constant = false;
    yyval.Value.type = new TypeInfo();
  ;
    break;}
case 243:
#line 1157 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Value.val = yyvsp[0].String;
    yyval.Value.constant = true;
    yyval.Value.type = new TypeInfo();
  ;
    break;}
case 244:
#line 1167 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    ResolveType(yyvsp[-1].Type);
    std::string Name = getUniqueName(yyvsp[0].Value.val, yyvsp[-1].Type);
    yyval.Value = yyvsp[0].Value;
    delete yyval.Value.val;
    delete yyval.Value.type;
    yyval.Value.val = new std::string(yyvsp[-1].Type->getNewTy() + " " + Name);
    yyval.Value.type = yyvsp[-1].Type;
  ;
    break;}
case 245:
#line 1177 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 246:
#line 1180 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ // Do not allow functions with 0 basic blocks   
    yyval.String = 0;
  ;
    break;}
case 247:
#line 1188 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 248:
#line 1192 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[0].String << '\n';
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 249:
#line 1197 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.String = 0;
  ;
    break;}
case 250:
#line 1200 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << *yyvsp[0].String << '\n';
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 252:
#line 1206 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = yyvsp[0].String; *yyval.String = "unwind"; ;
    break;}
case 253:
#line 1208 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{              // Return with a result...
    *O << "    " << *yyvsp[-1].String << ' ' << *yyvsp[0].Value.val << '\n';
    delete yyvsp[-1].String; yyvsp[0].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 254:
#line 1213 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                                       // Return with no result...
    *O << "    " << *yyvsp[-1].String << ' ' << yyvsp[0].Type->getNewTy() << '\n';
    delete yyvsp[-1].String; delete yyvsp[0].Type;
    yyval.String = 0;
  ;
    break;}
case 255:
#line 1218 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{                         // Unconditional Branch...
    *O << "    " << *yyvsp[-2].String << ' ' << yyvsp[-1].Type->getNewTy() << ' ' << *yyvsp[0].Value.val << '\n';
    delete yyvsp[-2].String; delete yyvsp[-1].Type; yyvsp[0].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 256:
#line 1223 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{  
    std::string Name = getUniqueName(yyvsp[-6].Value.val, yyvsp[-7].Type);
    *O << "    " << *yyvsp[-8].String << ' ' << yyvsp[-7].Type->getNewTy() << ' ' << Name << ", " 
       << yyvsp[-4].Type->getNewTy() << ' ' << *yyvsp[-3].Value.val << ", " << yyvsp[-1].Type->getNewTy() << ' ' 
       << *yyvsp[0].Value.val << '\n';
    delete yyvsp[-8].String; delete yyvsp[-7].Type; yyvsp[-6].Value.destroy(); delete yyvsp[-4].Type; yyvsp[-3].Value.destroy(); 
    delete yyvsp[-1].Type; yyvsp[0].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 257:
#line 1232 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[-6].Value.val, yyvsp[-7].Type);
    *O << "    " << *yyvsp[-8].String << ' ' << yyvsp[-7].Type->getNewTy() << ' ' << Name << ", " 
       << yyvsp[-4].Type->getNewTy() << ' ' << *yyvsp[-3].Value.val << " [" << *yyvsp[-1].String << " ]\n";
    delete yyvsp[-8].String; delete yyvsp[-7].Type; yyvsp[-6].Value.destroy(); delete yyvsp[-4].Type; yyvsp[-3].Value.destroy(); 
    delete yyvsp[-1].String;
    yyval.String = 0;
  ;
    break;}
case 258:
#line 1240 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[-5].Value.val, yyvsp[-6].Type);
    *O << "    " << *yyvsp[-7].String << ' ' << yyvsp[-6].Type->getNewTy() << ' ' << Name << ", " 
       << yyvsp[-3].Type->getNewTy() << ' ' << *yyvsp[-2].Value.val << "[]\n";
    delete yyvsp[-7].String; delete yyvsp[-6].Type; yyvsp[-5].Value.destroy(); delete yyvsp[-3].Type; yyvsp[-2].Value.destroy();
    yyval.String = 0;
  ;
    break;}
case 259:
#line 1248 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    TypeInfo* ResTy = getFunctionReturnType(yyvsp[-10].Type);
    *O << "    ";
    if (!yyvsp[-13].String->empty()) {
      std::string Name = getUniqueName(yyvsp[-13].String, ResTy);
      *O << Name << " = ";
    }
    *O << *yyvsp[-12].String << ' ' << *yyvsp[-11].String << ' ' << yyvsp[-10].Type->getNewTy() << ' ' << *yyvsp[-9].Value.val << " (";
    for (unsigned i = 0; i < yyvsp[-7].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[-7].ValList)[i];
      *O << *VI.val;
      if (i+1 < yyvsp[-7].ValList->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *yyvsp[-5].String << ' ' << yyvsp[-4].Type->getNewTy() << ' ' << *yyvsp[-3].Value.val << ' ' 
       << *yyvsp[-2].String << ' ' << yyvsp[-1].Type->getNewTy() << ' ' << *yyvsp[0].Value.val << '\n';
    delete yyvsp[-13].String; delete yyvsp[-12].String; delete yyvsp[-11].String; delete yyvsp[-10].Type; yyvsp[-9].Value.destroy(); delete yyvsp[-7].ValList; 
    delete yyvsp[-5].String; delete yyvsp[-4].Type; yyvsp[-3].Value.destroy(); delete yyvsp[-2].String; delete yyvsp[-1].Type; 
    yyvsp[0].Value.destroy(); 
    yyval.String = 0;
  ;
    break;}
case 260:
#line 1270 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[0].String << '\n';
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 261:
#line 1275 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *O << "    " << *yyvsp[0].String << '\n';
    delete yyvsp[0].String;
    yyval.String = 0;
  ;
    break;}
case 262:
#line 1281 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + yyvsp[-4].Type->getNewTy() + " " + *yyvsp[-3].String + ", " + yyvsp[-1].Type->getNewTy() + " " + 
           *yyvsp[0].Value.val;
    delete yyvsp[-4].Type; delete yyvsp[-3].String; delete yyvsp[-1].Type; yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-5].String;
  ;
    break;}
case 263:
#line 1287 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyvsp[-3].String->insert(0, yyvsp[-4].Type->getNewTy() + " " );
    *yyvsp[-3].String += ", " + yyvsp[-1].Type->getNewTy() + " " + *yyvsp[0].Value.val;
    delete yyvsp[-4].Type; delete yyvsp[-1].Type; yyvsp[0].Value.destroy();
    yyval.String = yyvsp[-3].String;
  ;
    break;}
case 264:
#line 1295 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-1].String->empty()) {
      if (deleteUselessCastFlag && *deleteUselessCastName == *yyvsp[-1].String) {
        *yyvsp[-1].String += " = ";
        yyvsp[-1].String->insert(0, "; "); // don't actually delete it, just comment it out
        delete deleteUselessCastName;
      } else {
        // Get a unique name for the name of this value, based on its type.
        *yyvsp[-1].String = getUniqueName(yyvsp[-1].String, yyvsp[0].Value.type) + " = ";
      }
    }
    *yyvsp[-1].String += *yyvsp[0].Value.val;
    yyvsp[0].Value.destroy();
    deleteUselessCastFlag = false;
    yyval.String = yyvsp[-1].String; 
  ;
    break;}
case 265:
#line 1313 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{    // Used for PHI nodes
    std::string Name = getUniqueName(yyvsp[-3].Value.val, yyvsp[-5].Type);
    Name.insert(0, yyvsp[-5].Type->getNewTy() + "[");
    Name += "," + *yyvsp[-1].Value.val + "]";
    yyval.Value.val = new std::string(Name);
    yyval.Value.type = yyvsp[-5].Type;
    yyvsp[-3].Value.destroy(); yyvsp[-1].Value.destroy();
  ;
    break;}
case 266:
#line 1321 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[-3].Value.val, yyvsp[-6].Value.type);
    *yyvsp[-6].Value.val += ", [" + Name + "," + *yyvsp[-1].Value.val + "]";
    yyvsp[-3].Value.destroy(); yyvsp[-1].Value.destroy();
    yyval.Value = yyvsp[-6].Value;
  ;
    break;}
case 267:
#line 1330 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.ValList = new ValueList();
    yyval.ValList->push_back(yyvsp[0].Value);
  ;
    break;}
case 268:
#line 1334 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.ValList = yyvsp[-2].ValList;
    yyval.ValList->push_back(yyvsp[0].Value);
  ;
    break;}
case 269:
#line 1341 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.ValList = yyvsp[0].ValList; ;
    break;}
case 270:
#line 1342 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.ValList = new ValueList(); ;
    break;}
case 271:
#line 1346 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].String;
    delete yyvsp[0].String;
    yyval.String = yyvsp[-1].String;
  ;
    break;}
case 273:
#line 1354 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* op = getDivRemOpcode(*yyvsp[-4].String, yyvsp[-3].Type); 
    std::string Name1 = getUniqueName(yyvsp[-2].Value.val, yyvsp[-3].Type);
    std::string Name2 = getUniqueName(yyvsp[0].Value.val, yyvsp[-3].Type);
    yyval.Value.val = new std::string(op);
    *yyval.Value.val += " " + yyvsp[-3].Type->getNewTy() + " " + Name1 + ", " + Name2;
    yyval.Value.type = yyvsp[-3].Type;
    delete yyvsp[-4].String; yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 274:
#line 1363 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name1 = getUniqueName(yyvsp[-2].Value.val, yyvsp[-3].Type);
    std::string Name2 = getUniqueName(yyvsp[0].Value.val, yyvsp[-3].Type);
    *yyvsp[-4].String += " " + yyvsp[-3].Type->getNewTy() + " " + Name1 + ", " + Name2;
    yyval.Value.val = yyvsp[-4].String;
    yyval.Value.type = yyvsp[-3].Type;
    yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 275:
#line 1371 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name1 = getUniqueName(yyvsp[-2].Value.val, yyvsp[-3].Type);
    std::string Name2 = getUniqueName(yyvsp[0].Value.val, yyvsp[-3].Type);
    *yyvsp[-4].String = getCompareOp(*yyvsp[-4].String, yyvsp[-3].Type);
    *yyvsp[-4].String += " " + yyvsp[-3].Type->getNewTy() + " " + Name1 + ", " + Name2;
    yyval.Value.val = yyvsp[-4].String;
    yyval.Value.type = new TypeInfo("bool",BoolTy);
    yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 276:
#line 1380 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name1 = getUniqueName(yyvsp[-2].Value.val, yyvsp[-3].Type);
    std::string Name2 = getUniqueName(yyvsp[0].Value.val, yyvsp[-3].Type);
    *yyvsp[-5].String += " " + *yyvsp[-4].String + " " + yyvsp[-3].Type->getNewTy() + " " + Name1 + "," + Name2;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = new TypeInfo("bool",BoolTy);
    delete yyvsp[-4].String; yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 277:
#line 1388 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name1 = getUniqueName(yyvsp[-2].Value.val, yyvsp[-3].Type);
    std::string Name2 = getUniqueName(yyvsp[0].Value.val, yyvsp[-3].Type);
    *yyvsp[-5].String += " " + *yyvsp[-4].String + " " + yyvsp[-3].Type->getNewTy() + " " + Name1 + "," + Name2;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = new TypeInfo("bool",BoolTy);
    delete yyvsp[-4].String; yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 278:
#line 1396 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    yyval.Value = yyvsp[0].Value;
    yyval.Value.val->insert(0, *yyvsp[-1].String + " ");
    delete yyvsp[-1].String;
  ;
    break;}
case 279:
#line 1401 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    const char* shiftop = yyvsp[-3].String->c_str();
    if (*yyvsp[-3].String == "shr")
      shiftop = (yyvsp[-2].Value.type->isUnsigned()) ? "lshr" : "ashr";
    yyval.Value.val = new std::string(shiftop);
    *yyval.Value.val += " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyval.Value.type = yyvsp[-2].Value.type;
    delete yyvsp[-3].String; delete yyvsp[-2].Value.val; yyvsp[0].Value.destroy();
  ;
    break;}
case 280:
#line 1410 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string source = *yyvsp[-2].Value.val;
    TypeInfo* SrcTy = yyvsp[-2].Value.type;
    TypeInfo* DstTy = ResolveType(yyvsp[0].Type);
    yyval.Value.val = new std::string();
    if (*yyvsp[-3].String == "cast") {
      *yyval.Value.val +=  getCastUpgrade(source, SrcTy, DstTy, false);
    } else {
      *yyval.Value.val += *yyvsp[-3].String + " " + source + " to " + DstTy->getNewTy();
    }
    yyval.Value.type = yyvsp[0].Type;
    // Check to see if this is a useless cast of a value to the same name
    // and the same type. Such casts will probably cause redefinition errors
    // when assembled and perform no code gen action so just remove them.
    if (*yyvsp[-3].String == "cast" || *yyvsp[-3].String == "bitcast")
      if (yyvsp[-2].Value.type->isInteger() && DstTy->isInteger() &&
          yyvsp[-2].Value.type->getBitWidth() == DstTy->getBitWidth()) {
        deleteUselessCastFlag = true; // Flag the "Inst" rule
        deleteUselessCastName = new std::string(*yyvsp[-2].Value.val); // save the name
        size_t pos = deleteUselessCastName->find_first_of("%\"",0);
        if (pos != std::string::npos) {
          // remove the type portion before val
          deleteUselessCastName->erase(0, pos);
        }
      }
    delete yyvsp[-3].String; yyvsp[-2].Value.destroy();
    delete yyvsp[-1].String;
  ;
    break;}
case 281:
#line 1438 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Value.val + ", " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = yyvsp[-2].Value.type;
    yyvsp[-4].Value.destroy(); delete yyvsp[-2].Value.val; yyvsp[0].Value.destroy();
  ;
    break;}
case 282:
#line 1444 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-3].String += " " + *yyvsp[-2].Value.val + ", " + yyvsp[0].Type->getNewTy();
    yyval.Value.val = yyvsp[-3].String;
    yyval.Value.type = yyvsp[0].Type;
    yyvsp[-2].Value.destroy();
  ;
    break;}
case 283:
#line 1450 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-3].String += " " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyval.Value.val = yyvsp[-3].String;
    ResolveType(yyvsp[-2].Value.type);
    yyval.Value.type = yyvsp[-2].Value.type->getElementType();
    delete yyvsp[-2].Value.val; yyvsp[0].Value.destroy();
  ;
    break;}
case 284:
#line 1457 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Value.val + ", " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = yyvsp[-4].Value.type;
    delete yyvsp[-4].Value.val; yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 285:
#line 1463 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-5].String += " " + *yyvsp[-4].Value.val + ", " + *yyvsp[-2].Value.val + ", " + *yyvsp[0].Value.val;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = yyvsp[-4].Value.type;
    delete yyvsp[-4].Value.val; yyvsp[-2].Value.destroy(); yyvsp[0].Value.destroy();
  ;
    break;}
case 286:
#line 1469 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].Value.val;
    yyval.Value.val = yyvsp[-1].String;
    yyval.Value.type = yyvsp[0].Value.type;
    delete yyvsp[0].Value.val;
  ;
    break;}
case 287:
#line 1475 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    if (!yyvsp[-5].String->empty())
      *yyvsp[-6].String += " " + *yyvsp[-5].String;
    if (!yyvsp[-6].String->empty())
      *yyvsp[-6].String += " ";
    *yyvsp[-6].String += yyvsp[-4].Type->getNewTy() + " " + *yyvsp[-3].Value.val + "(";
    for (unsigned i = 0; i < yyvsp[-1].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[-1].ValList)[i];
      *yyvsp[-6].String += *VI.val;
      if (i+1 < yyvsp[-1].ValList->size())
        *yyvsp[-6].String += ", ";
      VI.destroy();
    }
    *yyvsp[-6].String += ")";
    yyval.Value.val = yyvsp[-6].String;
    yyval.Value.type = getFunctionReturnType(yyvsp[-4].Type);
    delete yyvsp[-5].String; delete yyvsp[-4].Type; yyvsp[-3].Value.destroy(); delete yyvsp[-1].ValList;
  ;
    break;}
case 289:
#line 1498 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.ValList = yyvsp[0].ValList; ;
    break;}
case 290:
#line 1499 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{  yyval.ValList = new ValueList(); ;
    break;}
case 292:
#line 1504 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{ yyval.String = new std::string(); ;
    break;}
case 293:
#line 1507 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " " + yyvsp[-1].Type->getNewTy();
    if (!yyvsp[0].String->empty())
      *yyvsp[-2].String += " " + *yyvsp[0].String;
    yyval.Value.val = yyvsp[-2].String;
    yyval.Value.type = yyvsp[-1].Type->getPointerType();
    delete yyvsp[-1].Type; delete yyvsp[0].String;
  ;
    break;}
case 294:
#line 1515 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[-1].Value.val, yyvsp[-2].Type);
    *yyvsp[-5].String += " " + yyvsp[-4].Type->getNewTy() + ", " + yyvsp[-2].Type->getNewTy() + " " + Name;
    if (!yyvsp[0].String->empty())
      *yyvsp[-5].String += " " + *yyvsp[0].String;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = yyvsp[-4].Type->getPointerType();
    delete yyvsp[-4].Type; delete yyvsp[-2].Type; yyvsp[-1].Value.destroy(); delete yyvsp[0].String;
  ;
    break;}
case 295:
#line 1524 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-2].String += " " + yyvsp[-1].Type->getNewTy();
    if (!yyvsp[0].String->empty())
      *yyvsp[-2].String += " " + *yyvsp[0].String;
    yyval.Value.val = yyvsp[-2].String;
    yyval.Value.type = yyvsp[-1].Type->getPointerType();
    delete yyvsp[-1].Type; delete yyvsp[0].String;
  ;
    break;}
case 296:
#line 1532 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[-1].Value.val, yyvsp[-2].Type);
    *yyvsp[-5].String += " " + yyvsp[-4].Type->getNewTy() + ", " + yyvsp[-2].Type->getNewTy() + " " + Name;
    if (!yyvsp[0].String->empty())
      *yyvsp[-5].String += " " + *yyvsp[0].String;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = yyvsp[-4].Type->getPointerType();
    delete yyvsp[-4].Type; delete yyvsp[-2].Type; yyvsp[-1].Value.destroy(); delete yyvsp[0].String;
  ;
    break;}
case 297:
#line 1541 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    *yyvsp[-1].String += " " + *yyvsp[0].Value.val;
    yyval.Value.val = yyvsp[-1].String;
    yyval.Value.type = new TypeInfo("void", VoidTy); 
    yyvsp[0].Value.destroy();
  ;
    break;}
case 298:
#line 1547 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[0].Value.val, yyvsp[-1].Type);
    if (!yyvsp[-3].String->empty())
      *yyvsp[-3].String += " ";
    *yyvsp[-3].String += *yyvsp[-2].String + " " + yyvsp[-1].Type->getNewTy() + " " + Name;
    yyval.Value.val = yyvsp[-3].String;
    yyval.Value.type = yyvsp[-1].Type->getElementType()->clone();
    delete yyvsp[-2].String; delete yyvsp[-1].Type; yyvsp[0].Value.destroy();
  ;
    break;}
case 299:
#line 1556 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[0].Value.val, yyvsp[-1].Type);
    if (!yyvsp[-5].String->empty())
      *yyvsp[-5].String += " ";
    *yyvsp[-5].String += *yyvsp[-4].String + " " + *yyvsp[-3].Value.val + ", " + yyvsp[-1].Type->getNewTy() + " " + Name;
    yyval.Value.val = yyvsp[-5].String;
    yyval.Value.type = new TypeInfo("void", VoidTy);
    delete yyvsp[-4].String; yyvsp[-3].Value.destroy(); delete yyvsp[-1].Type; yyvsp[0].Value.destroy();
  ;
    break;}
case 300:
#line 1565 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"
{
    std::string Name = getUniqueName(yyvsp[-1].Value.val, yyvsp[-2].Type);
    // Upgrade the indices
    for (unsigned i = 0; i < yyvsp[0].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[0].ValList)[i];
      if (VI.type->isUnsigned() && !VI.isConstant() && 
          VI.type->getBitWidth() < 64) {
        std::string* old = VI.val;
        *O << "    %gep_upgrade" << unique << " = zext " << *old 
           << " to i64\n";
        VI.val = new std::string("i64 %gep_upgrade" + llvm::utostr(unique++));
        VI.type->setOldTy(ULongTy);
      }
    }
    *yyvsp[-3].String += " " + yyvsp[-2].Type->getNewTy() + " " + Name;
    for (unsigned i = 0; i < yyvsp[0].ValList->size(); ++i) {
      ValueInfo& VI = (*yyvsp[0].ValList)[i];
      *yyvsp[-3].String += ", " + *VI.val;
    }
    yyval.Value.val = yyvsp[-3].String;
    yyval.Value.type = getGEPIndexedType(yyvsp[-2].Type,yyvsp[0].ValList); 
    yyvsp[-1].Value.destroy(); delete yyvsp[0].ValList;
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
#line 1589 "/Volumes/ProjectsDisk/cvs/llvm/tools/llvm-upgrade/UpgradeParser.y"


int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
  *O << "llvm-upgrade parse failed.\n";
  exit(1);
}
