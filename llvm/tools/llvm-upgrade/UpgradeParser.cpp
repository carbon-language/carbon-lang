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
     CAST = 367,
     TRUNC = 368,
     ZEXT = 369,
     SEXT = 370,
     FPTRUNC = 371,
     FPEXT = 372,
     FPTOUI = 373,
     FPTOSI = 374,
     UITOFP = 375,
     SITOFP = 376,
     PTRTOINT = 377,
     INTTOPTR = 378,
     BITCAST = 379
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
#define TRUNC 368
#define ZEXT 369
#define SEXT 370
#define FPTRUNC 371
#define FPEXT 372
#define FPTOUI 373
#define FPTOSI 374
#define UITOFP 375
#define SITOFP 376
#define PTRTOINT 377
#define INTTOPTR 378
#define BITCAST 379




/* Copy the first part of user declarations.  */
#line 14 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"

#include "ParserInternals.h"
#include <llvm/ADT/StringExtras.h>
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

typedef std::vector<TypeInfo> TypeVector;
static TypeVector EnumeratedTypes;
typedef std::map<std::string,TypeInfo> TypeMap;
static TypeMap NamedTypes;

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
    if (I != NamedTypes.end())
      Ty.oldTy = I->second.oldTy;
    else {
      std::string msg("Can't resolve type: ");
      msg += *Ty.newTy;
      yyerror(msg.c_str());
    }
  } else if (Ty.oldTy == NumericTy) {
    unsigned ref = atoi(&((Ty.newTy->c_str())[1])); // Skip the '\\'
    if (ref < EnumeratedTypes.size()) {
      Ty.oldTy = EnumeratedTypes[ref].oldTy;
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
      *O << "    %cast_upgrade = fptoui " + Source + " to ulong\n";
      Source = "ulong %cast_upgrade";
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
#line 209 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
typedef union YYSTYPE {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
} YYSTYPE;
/* Line 196 of yacc.c.  */
#line 542 "UpgradeParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 554 "UpgradeParser.tab.c"

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
#define YYLAST   1335

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  139
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  71
/* YYNRULES -- Number of rules. */
#define YYNRULES  263
/* YYNRULES -- Number of states. */
#define YYNSTATES  522

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   379

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     128,   129,   137,     2,   126,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     133,   125,   134,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   130,   127,   132,     2,     2,     2,     2,     2,   138,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     131,     2,     2,   135,     2,   136,     2,     2,     2,     2,
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
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124
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
      79,    81,    83,    85,    87,    89,    91,    93,    95,    97,
      99,   101,   103,   105,   108,   109,   111,   113,   115,   117,
     119,   121,   123,   124,   126,   128,   130,   132,   134,   136,
     139,   140,   141,   144,   145,   149,   152,   153,   155,   156,
     160,   162,   165,   167,   169,   171,   173,   175,   177,   179,
     181,   183,   185,   187,   189,   191,   193,   195,   197,   199,
     201,   203,   205,   208,   213,   219,   225,   229,   232,   235,
     237,   241,   243,   247,   249,   250,   255,   259,   263,   268,
     273,   277,   280,   283,   286,   289,   292,   295,   298,   301,
     304,   307,   314,   320,   329,   336,   343,   350,   357,   364,
     373,   382,   386,   388,   390,   392,   394,   397,   400,   405,
     408,   410,   415,   418,   423,   430,   437,   444,   451,   455,
     460,   461,   463,   465,   467,   471,   475,   479,   483,   487,
     491,   493,   494,   496,   498,   500,   501,   504,   508,   510,
     512,   516,   518,   519,   528,   530,   532,   536,   538,   540,
     544,   545,   547,   549,   553,   554,   556,   558,   560,   562,
     564,   566,   568,   570,   572,   576,   578,   584,   586,   588,
     590,   592,   595,   598,   600,   603,   606,   607,   609,   612,
     615,   619,   629,   639,   648,   663,   665,   667,   674,   680,
     683,   690,   698,   700,   704,   706,   707,   710,   712,   718,
     724,   730,   733,   738,   743,   750,   755,   760,   767,   774,
     777,   785,   787,   790,   791,   793,   794,   798,   805,   809,
     816,   819,   824,   831
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     171,     0,    -1,    19,    -1,    20,    -1,    17,    -1,    18,
      -1,    78,    -1,    79,    -1,    80,    -1,    81,    -1,    82,
      -1,    83,    -1,    84,    -1,    85,    -1,    86,    -1,    87,
      -1,    88,    -1,    89,    -1,    90,    -1,    91,    -1,    92,
      -1,    93,    -1,    94,    -1,    95,    -1,   104,    -1,   105,
      -1,   106,    -1,   107,    -1,   113,    -1,   114,    -1,   115,
      -1,   116,    -1,   117,    -1,   118,    -1,   119,    -1,   120,
      -1,   121,    -1,   122,    -1,   123,    -1,   124,    -1,   112,
      -1,    11,    -1,     9,    -1,     7,    -1,     5,    -1,    12,
      -1,    10,    -1,     8,    -1,     6,    -1,   147,    -1,   148,
      -1,    13,    -1,    14,    -1,   179,   125,    -1,    -1,    42,
      -1,    43,    -1,    44,    -1,    48,    -1,    45,    -1,    46,
      -1,    47,    -1,    -1,    65,    -1,    66,    -1,    67,    -1,
      68,    -1,    69,    -1,    70,    -1,    64,    18,    -1,    -1,
      -1,    57,    18,    -1,    -1,   126,    57,    18,    -1,    37,
      30,    -1,    -1,   156,    -1,    -1,   126,   159,   158,    -1,
     156,    -1,    57,    18,    -1,   162,    -1,     3,    -1,   164,
      -1,     3,    -1,   164,    -1,     4,    -1,     5,    -1,     6,
      -1,     7,    -1,     8,    -1,     9,    -1,    10,    -1,    11,
      -1,    12,    -1,    13,    -1,    14,    -1,    15,    -1,    16,
      -1,   193,    -1,   163,    -1,   127,    18,    -1,   161,   128,
     166,   129,    -1,   130,    18,   131,   164,   132,    -1,   133,
      18,   131,   164,   134,    -1,   135,   165,   136,    -1,   135,
     136,    -1,   164,   137,    -1,   164,    -1,   165,   126,   164,
      -1,   165,    -1,   165,   126,    40,    -1,    40,    -1,    -1,
     162,   130,   169,   132,    -1,   162,   130,   132,    -1,   162,
     138,    30,    -1,   162,   133,   169,   134,    -1,   162,   135,
     169,   136,    -1,   162,   135,   136,    -1,   162,    22,    -1,
     162,    23,    -1,   162,   193,    -1,   162,   168,    -1,   162,
      24,    -1,   147,   141,    -1,   148,    18,    -1,     4,    25,
      -1,     4,    26,    -1,   150,    21,    -1,   146,   128,   167,
      39,   162,   129,    -1,   101,   128,   167,   207,   129,    -1,
     103,   128,   167,   126,   167,   126,   167,   129,    -1,   142,
     128,   167,   126,   167,   129,    -1,   143,   128,   167,   126,
     167,   129,    -1,   144,   128,   167,   126,   167,   129,    -1,
     145,   128,   167,   126,   167,   129,    -1,   109,   128,   167,
     126,   167,   129,    -1,   110,   128,   167,   126,   167,   126,
     167,   129,    -1,   111,   128,   167,   126,   167,   126,   167,
     129,    -1,   169,   126,   167,    -1,   167,    -1,    35,    -1,
      36,    -1,   172,    -1,   172,   188,    -1,   172,   190,    -1,
     172,    62,    61,   174,    -1,   172,    31,    -1,   173,    -1,
     173,   151,    27,   160,    -1,   173,   190,    -1,   173,    62,
      61,   174,    -1,   173,   151,   152,   170,   167,   158,    -1,
     173,   151,    50,   170,   162,   158,    -1,   173,   151,    45,
     170,   162,   158,    -1,   173,   151,    47,   170,   162,   158,
      -1,   173,    51,   176,    -1,   173,    58,   125,   177,    -1,
      -1,    30,    -1,    56,    -1,    55,    -1,    53,   125,   175,
      -1,    54,   125,    18,    -1,    52,   125,    30,    -1,    71,
     125,    30,    -1,   130,   178,   132,    -1,   178,   126,    30,
      -1,    30,    -1,    -1,    28,    -1,    30,    -1,   179,    -1,
      -1,   162,   180,    -1,   182,   126,   181,    -1,   181,    -1,
     182,    -1,   182,   126,    40,    -1,    40,    -1,    -1,   153,
     160,   179,   128,   183,   129,   157,   154,    -1,    32,    -1,
     135,    -1,   152,   184,   185,    -1,    33,    -1,   136,    -1,
     186,   196,   187,    -1,    -1,    45,    -1,    47,    -1,    34,
     189,   184,    -1,    -1,    63,    -1,    17,    -1,    18,    -1,
      21,    -1,    25,    -1,    26,    -1,    22,    -1,    23,    -1,
      24,    -1,   133,   169,   134,    -1,   168,    -1,    61,   191,
      30,   126,    30,    -1,   140,    -1,   179,    -1,   193,    -1,
     192,    -1,   162,   194,    -1,   196,   197,    -1,   197,    -1,
     198,   199,    -1,   198,   201,    -1,    -1,    29,    -1,    72,
     195,    -1,    72,     3,    -1,    73,    15,   194,    -1,    73,
       4,   194,   126,    15,   194,   126,    15,   194,    -1,    74,
     149,   194,   126,    15,   194,   130,   200,   132,    -1,    74,
     149,   194,   126,    15,   194,   130,   132,    -1,   151,    75,
     153,   160,   194,   128,   204,   129,    39,    15,   194,    76,
      15,   194,    -1,    76,    -1,    77,    -1,   200,   149,   192,
     126,    15,   194,    -1,   149,   192,   126,    15,   194,    -1,
     151,   206,    -1,   162,   130,   194,   126,   194,   132,    -1,
     202,   126,   130,   194,   126,   194,   132,    -1,   195,    -1,
     203,   126,   195,    -1,   203,    -1,    -1,    60,    59,    -1,
      59,    -1,   142,   162,   194,   126,   194,    -1,   143,   162,
     194,   126,   194,    -1,   144,   162,   194,   126,   194,    -1,
      49,   195,    -1,   145,   195,   126,   195,    -1,   146,   195,
      39,   162,    -1,   103,   195,   126,   195,   126,   195,    -1,
     108,   195,   126,   162,    -1,   109,   195,   126,   195,    -1,
     110,   195,   126,   195,   126,   195,    -1,   111,   195,   126,
     195,   126,   195,    -1,   102,   202,    -1,   205,   153,   160,
     194,   128,   204,   129,    -1,   209,    -1,   126,   203,    -1,
      -1,    38,    -1,    -1,    96,   162,   155,    -1,    96,   162,
     126,    10,   194,   155,    -1,    97,   162,   155,    -1,    97,
     162,   126,    10,   194,   155,    -1,    98,   195,    -1,   208,
      99,   162,   194,    -1,   208,   100,   195,   126,   162,   194,
      -1,   101,   162,   194,   207,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   269,   269,   269,   270,   270,   274,   274,   274,   274,
     274,   274,   274,   274,   274,   275,   275,   275,   276,   276,
     276,   276,   276,   276,   277,   277,   277,   277,   278,   278,
     278,   278,   278,   278,   278,   279,   279,   279,   279,   279,
     279,   284,   284,   284,   284,   285,   285,   285,   285,   286,
     286,   287,   287,   290,   293,   298,   298,   298,   298,   298,
     298,   299,   300,   303,   303,   303,   303,   303,   304,   305,
     310,   315,   316,   319,   320,   328,   334,   335,   338,   339,
     348,   349,   362,   362,   363,   363,   364,   368,   368,   368,
     368,   368,   368,   368,   369,   369,   369,   369,   369,   371,
     375,   379,   384,   389,   395,   402,   409,   415,   419,   429,
     432,   440,   441,   446,   449,   459,   465,   470,   476,   482,
     488,   493,   499,   505,   511,   517,   523,   529,   535,   541,
     547,   555,   569,   581,   586,   591,   596,   601,   609,   614,
     619,   629,   634,   639,   639,   649,   654,   657,   662,   666,
     670,   673,   684,   689,   694,   701,   708,   715,   722,   727,
     732,   737,   739,   739,   742,   747,   754,   759,   766,   773,
     778,   779,   787,   787,   788,   788,   790,   797,   801,   805,
     808,   813,   816,   818,   838,   841,   845,   854,   855,   857,
     865,   866,   867,   871,   884,   885,   888,   888,   888,   888,
     888,   888,   888,   889,   890,   895,   896,   905,   905,   908,
     908,   914,   921,   924,   932,   936,   941,   944,   950,   955,
     960,   965,   972,   978,   984,  1004,  1009,  1015,  1020,  1028,
    1037,  1043,  1051,  1055,  1062,  1063,  1067,  1072,  1075,  1080,
    1085,  1090,  1095,  1103,  1117,  1122,  1127,  1132,  1137,  1142,
    1147,  1164,  1169,  1170,  1174,  1175,  1178,  1185,  1192,  1199,
    1206,  1211,  1218,  1225
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
  "SHUFFLEVECTOR", "CAST", "TRUNC", "ZEXT", "SEXT", "FPTRUNC", "FPEXT",
  "FPTOUI", "FPTOSI", "UITOFP", "SITOFP", "PTRTOINT", "INTTOPTR",
  "BITCAST", "'='", "','", "'\\\\'", "'('", "')'", "'['", "'x'", "']'",
  "'<'", "'>'", "'{'", "'}'", "'*'", "'c'", "$accept", "IntVal",
  "EInt64Val", "ArithmeticOps", "LogicalOps", "SetCondOps", "ShiftOps",
  "CastOps", "SIntType", "UIntType", "IntType", "FPType", "OptAssign",
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
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,    61,    44,    92,    40,    41,
      91,   120,    93,    60,    62,   123,   125,    42,    99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   139,   140,   140,   141,   141,   142,   142,   142,   142,
     142,   142,   142,   142,   142,   143,   143,   143,   144,   144,
     144,   144,   144,   144,   145,   145,   145,   145,   146,   146,
     146,   146,   146,   146,   146,   146,   146,   146,   146,   146,
     146,   147,   147,   147,   147,   148,   148,   148,   148,   149,
     149,   150,   150,   151,   151,   152,   152,   152,   152,   152,
     152,   152,   152,   153,   153,   153,   153,   153,   153,   153,
     153,   154,   154,   155,   155,   156,   157,   157,   158,   158,
     159,   159,   160,   160,   161,   161,   162,   163,   163,   163,
     163,   163,   163,   163,   163,   163,   163,   163,   163,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   165,
     165,   166,   166,   166,   166,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     168,   169,   169,   170,   170,   171,   172,   172,   172,   172,
     172,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   174,   175,   175,   176,   176,   176,   176,   177,   178,
     178,   178,   179,   179,   180,   180,   181,   182,   182,   183,
     183,   183,   183,   184,   185,   185,   186,   187,   187,   188,
     189,   189,   189,   190,   191,   191,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   193,   193,   194,
     194,   195,   196,   196,   197,   198,   198,   198,   199,   199,
     199,   199,   199,   199,   199,   199,   199,   200,   200,   201,
     202,   202,   203,   203,   204,   204,   205,   205,   206,   206,
     206,   206,   206,   206,   206,   206,   206,   206,   206,   206,
     206,   206,   207,   207,   208,   208,   209,   209,   209,   209,
     209,   209,   209,   209
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     0,     1,     1,     1,     1,     1,
       1,     1,     0,     1,     1,     1,     1,     1,     1,     2,
       0,     0,     2,     0,     3,     2,     0,     1,     0,     3,
       1,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     4,     5,     5,     3,     2,     2,     1,
       3,     1,     3,     1,     0,     4,     3,     3,     4,     4,
       3,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     6,     5,     8,     6,     6,     6,     6,     6,     8,
       8,     3,     1,     1,     1,     1,     2,     2,     4,     2,
       1,     4,     2,     4,     6,     6,     6,     6,     3,     4,
       0,     1,     1,     1,     3,     3,     3,     3,     3,     3,
       1,     0,     1,     1,     1,     0,     2,     3,     1,     1,
       3,     1,     0,     8,     1,     1,     3,     1,     1,     3,
       0,     1,     1,     3,     0,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     1,     5,     1,     1,     1,
       1,     2,     2,     1,     2,     2,     0,     1,     2,     2,
       3,     9,     9,     8,    14,     1,     1,     6,     5,     2,
       6,     7,     1,     3,     1,     0,     2,     1,     5,     5,
       5,     2,     4,     4,     6,     4,     4,     6,     6,     2,
       7,     1,     2,     0,     1,     0,     3,     6,     3,     6,
       2,     4,     6,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned short int yydefact[] =
{
     160,     0,    62,   150,     1,   149,   190,    55,    56,    57,
      59,    60,    61,    58,     0,    70,   216,   146,   147,   172,
     173,     0,     0,     0,    62,     0,   152,   191,   192,    70,
       0,     0,    63,    64,    65,    66,    67,    68,     0,     0,
     217,   216,   213,    54,     0,     0,     0,     0,   158,     0,
       0,     0,     0,     0,     0,     0,    53,   193,   161,   148,
      69,    83,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,     2,     3,     0,     0,     0,
       0,   207,     0,     0,    82,   101,    86,   208,   100,   184,
     185,   186,   187,   188,   189,   212,     0,     0,     0,   225,
     226,   255,   214,   215,     0,     0,     0,     0,   171,   159,
     153,   151,   143,   144,     0,     0,     0,     0,   102,     0,
       0,    85,   107,   109,     0,     0,   114,   108,   219,     0,
     218,     0,     0,    44,    48,    43,    47,    42,    46,    41,
      45,    49,    50,     0,   254,     0,   237,     0,    70,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,     0,     0,     0,
       0,     0,     0,    24,    25,    26,    27,     0,     0,     0,
       0,    40,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,     0,     0,     0,     0,     0,    70,
     229,     0,   251,   166,   163,   162,   164,   165,   167,   170,
       0,    78,    78,    78,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,     0,     0,     0,     0,    78,
       0,     0,     0,   106,   182,   113,   111,     0,   196,   197,
     198,   201,   202,   203,   199,   200,   194,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   205,   210,
     209,   211,     0,   220,     0,   241,   236,     0,    73,    73,
     260,     0,     0,   249,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   168,     0,
     156,   157,   155,   128,   129,     4,     5,   126,   127,   130,
     121,   122,   125,     0,     0,     0,     0,   124,   123,   154,
      84,    84,   110,   181,   175,   178,   179,     0,     0,   103,
     195,     0,     0,     0,     0,     0,     0,   142,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   256,     0,
     258,   253,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   169,     0,     0,
      80,    78,   116,     0,     0,   120,     0,   117,   104,   105,
     174,   176,     0,    76,   112,     0,   253,     0,     0,     0,
       0,     0,   204,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   263,     0,     0,     0,   245,
     246,     0,     0,     0,     0,     0,   242,   243,     0,   261,
       0,    75,    81,    79,   115,   118,   119,   180,   177,    77,
      71,     0,     0,     0,     0,     0,     0,   141,     0,     0,
       0,     0,     0,     0,     0,   235,    73,    74,    73,   232,
     252,     0,     0,     0,     0,     0,   238,   239,   240,   235,
       0,     0,   183,   206,   132,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   234,     0,     0,   257,
     259,     0,     0,     0,   244,   247,   248,     0,   262,    72,
       0,   138,     0,     0,   134,   135,   136,   137,   131,     0,
     223,     0,     0,     0,   233,   230,     0,   250,     0,     0,
       0,   221,     0,   222,     0,     0,   231,   133,   139,   140,
       0,     0,     0,     0,     0,     0,   228,     0,     0,   227,
       0,   224
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,    81,   297,   253,   254,   255,   256,   257,   225,   226,
     143,   227,    24,    15,    38,   452,   338,   360,   420,   290,
     361,    82,    83,   228,    85,    86,   124,   237,   327,   258,
     328,   114,     1,     2,     3,    59,   206,    48,   109,   210,
      87,   371,   315,   316,   317,    39,    91,    16,    94,    17,
      29,    18,   321,   259,    88,   261,   439,    41,    42,    43,
     102,   492,   103,   273,   466,   467,   199,   200,   395,   201,
     202
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -472
static const short int yypact[] =
{
    -472,    38,   138,  1015,  -472,  -472,    59,  -472,  -472,  -472,
    -472,  -472,  -472,  -472,    -2,    80,    35,  -472,  -472,  -472,
    -472,   -29,   -50,    18,   174,   -37,  -472,  -472,  -472,    80,
      73,    93,  -472,  -472,  -472,  -472,  -472,  -472,   744,   -22,
    -472,   -21,  -472,     9,     3,    12,    54,    64,  -472,    37,
      73,   744,    84,    84,    84,    84,  -472,  -472,  -472,  -472,
    -472,    62,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
    -472,  -472,  -472,  -472,  -472,  -472,  -472,   173,   176,   177,
     431,  -472,    86,    69,  -472,  -472,  -106,  -472,  -472,  -472,
    -472,  -472,  -472,  -472,  -472,  -472,   772,    28,   343,  -472,
    -472,  1211,  -472,  -472,   162,    66,   180,   169,   172,  -472,
    -472,  -472,  -472,  -472,   802,   802,   802,   831,  -472,    72,
      74,  -472,  -472,  -106,   -46,    76,   504,  -472,    62,  1011,
    -472,  1011,  1011,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
    -472,  -472,  -472,  1011,  -472,   802,  -472,   148,    80,  -472,
    -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
    -472,  -472,  -472,  -472,  -472,  -472,  -472,   802,   802,   802,
     802,   802,   802,  -472,  -472,  -472,  -472,   802,   802,   802,
     802,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
    -472,  -472,  -472,  -472,   802,   802,   802,   802,   802,    80,
    -472,    24,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
     -70,    83,    83,    83,   101,   118,   192,   137,   193,   147,
     194,   153,   195,   204,   207,   170,   197,   209,   889,    83,
     802,   802,   802,  -472,   565,  -472,   105,   103,  -472,  -472,
    -472,  -472,  -472,  -472,  -472,  -472,   171,   107,   108,   109,
     110,   113,   831,   114,   115,   116,   117,   122,  -472,  -472,
    -472,  -472,   127,  -472,   128,  -472,  -472,   744,   129,   130,
    -472,  1011,   132,   131,   133,   144,   145,   152,   154,  1011,
    1011,  1011,   155,   219,   744,   802,   802,   243,  -472,   -16,
    -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
    -472,  -472,  -472,   610,   831,   476,   244,  -472,  -472,  -472,
    -102,   -32,  -106,  -472,    86,  -472,   156,   150,   638,  -472,
    -472,   253,   831,   831,   831,   831,   831,  -472,   -90,   831,
     831,   831,   831,   831,   269,   272,  1011,     4,  -472,    17,
    -472,   163,  1011,   161,   802,   802,   802,   802,   802,   166,
     167,   178,   802,   802,  1011,  1011,   179,  -472,   265,   278,
    -472,    83,  -472,   -39,   -71,  -472,   -35,  -472,  -472,  -472,
    -472,  -472,   698,   260,  -472,   185,   163,   186,   203,   205,
     206,   831,  -472,   210,   211,   213,   214,   259,  1011,  1011,
     175,  1011,   281,  1011,   802,  -472,   215,  1011,   216,  -472,
    -472,   217,   218,  1011,  1011,  1011,  -472,  -472,   202,  -472,
     802,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,  -472,
     266,   303,   228,   831,   831,   831,   831,  -472,   831,   831,
     831,   831,   802,   220,   229,   802,   232,  -472,   232,  -472,
     234,  1011,   236,   802,   802,   802,  -472,  -472,  -472,   802,
    1011,   317,  -472,  -472,  -472,   240,   238,   245,   247,   246,
     248,   249,   250,   251,   359,    42,   234,   252,   325,  -472,
    -472,   802,   255,  1011,  -472,  -472,  -472,   254,  -472,  -472,
     831,  -472,   831,   831,  -472,  -472,  -472,  -472,  -472,  1011,
    -472,  1124,    61,   349,  -472,  -472,   258,  -472,   264,   270,
     271,  -472,   275,  -472,  1124,   383,  -472,  -472,  -472,  -472,
     387,   277,  1011,  1011,   389,   332,  -472,  1011,   397,  -472,
    1011,  -472
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -472,  -472,  -472,   312,   314,   315,   321,   326,   -97,   -96,
    -446,  -472,   385,   405,  -139,  -472,  -263,    57,  -472,  -195,
    -472,   -44,  -472,   -38,  -472,   -69,   305,  -472,    -5,   221,
    -209,    55,  -472,  -472,  -472,   382,  -472,  -472,  -472,  -472,
       2,  -472,    81,  -472,  -472,   419,  -472,  -472,  -472,  -472,
    -472,   449,  -472,  -471,  -103,  -128,   -80,  -472,   413,  -472,
    -472,  -472,  -472,  -472,    63,     6,  -472,  -472,    82,  -472,
    -472
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -146
static const short int yytable[] =
{
      84,   141,   142,   262,   263,    25,   340,   111,    40,   267,
      89,   123,    92,    84,   391,   264,   130,   291,   292,   491,
     502,   358,   -84,    44,    45,    46,   260,   393,   260,   260,
     368,   127,   131,   511,   309,   127,   381,    19,     4,    20,
     260,   359,    47,   132,   382,    25,   504,   133,   134,   135,
     136,   137,   138,   139,   140,   381,   287,   123,   129,    30,
     284,   392,   288,   415,    40,   265,   133,   134,   135,   136,
     137,   138,   139,   140,   392,    49,   211,   212,   213,    50,
     232,    96,    97,    98,   125,    99,   100,   381,    56,   270,
     233,   381,   274,   414,   363,   364,   366,   275,   276,   277,
     278,   416,   369,    58,    27,   127,    28,   129,   115,   116,
     117,    60,   229,    90,    19,    93,    20,   282,   283,   112,
     113,   204,   205,   285,   286,   308,   293,   294,   104,   268,
     269,   129,   271,   272,   129,   -44,   -44,   105,  -145,   129,
     129,   129,   129,   341,    31,    32,    33,    34,    35,    36,
      37,   349,   350,   351,   -43,   -43,   279,   280,   281,   129,
     129,   310,   311,   312,   -42,   -42,   413,   108,   260,     5,
     -41,   -41,     6,   469,   490,   470,   260,   260,   260,   106,
       7,     8,     9,    10,    11,    12,    13,   295,   296,   107,
     -85,   118,   203,   503,   119,   120,   314,   126,   207,   208,
      14,    51,   209,   230,   234,   231,   356,   266,   390,   289,
     -48,   -47,   -46,   -45,   396,   298,     7,     8,     9,    52,
      11,    53,    13,   336,    54,   -51,   408,   409,   -52,    84,
     299,   318,   319,   260,   320,   322,   323,   324,   325,   260,
     354,   326,   329,   330,   331,   332,    84,   355,   129,   312,
     333,   260,   260,   334,   335,   337,   339,   343,   353,   344,
     433,   434,   342,   436,   398,   438,   400,   401,   402,   442,
     345,   346,   406,   357,   367,   446,   447,   448,   347,   373,
     348,   352,   372,   375,   388,   260,   260,   389,   260,   394,
     260,   397,   403,   404,   260,   411,   412,   358,   432,   437,
     260,   260,   260,   435,   405,   410,   129,   399,   129,   129,
     129,   421,   423,   472,   129,   407,   370,   376,   377,   378,
     379,   380,   478,   451,   383,   384,   385,   386,   387,   424,
     449,   425,   426,   453,   314,   479,   428,   429,   260,   430,
     431,   441,   443,   444,   445,   496,   464,   260,   133,   134,
     135,   136,   137,   138,   139,   140,   129,   454,   468,   465,
     471,   501,   473,   474,   475,   476,   480,   481,   141,   142,
     260,   482,   450,   483,   489,   484,   427,   485,   486,   487,
     488,   493,   392,   497,   515,   516,   260,   495,   505,   519,
     506,   494,   521,   507,   463,   141,   142,   129,   512,   508,
     509,   510,   513,   514,   517,   129,   129,   129,   518,   260,
     260,   129,   520,   194,   260,   195,   196,   260,   455,   456,
     457,   458,   197,   459,   460,   461,   462,   198,   101,    55,
     419,   236,   110,   129,   121,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    57,   307,
      75,    76,    26,   418,    95,   477,     0,   440,   422,    19,
       0,    20,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   498,     0,   499,   500,   121,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     224,    73,    74,     0,     0,    75,    76,     0,     0,     0,
       0,     0,     0,     0,    19,     0,    20,   121,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,     0,     0,    75,    76,     0,     0,     0,     0,     0,
       0,     0,    19,     0,    20,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   235,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    77,     0,
       0,    78,     0,     0,    79,     0,    80,   122,   121,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,     0,     0,    75,    76,     0,     0,     0,     0,
       0,     0,     0,    19,     0,    20,     0,     0,     0,     0,
       0,     0,     0,    77,     0,   313,    78,     0,     0,    79,
       0,    80,   365,   121,   214,   215,   216,   217,   218,   219,
     220,   221,   222,   223,   224,    73,    74,     0,     0,    75,
      76,    77,     0,     0,    78,     0,     0,    79,    19,    80,
      20,   121,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,     0,     0,    75,    76,     0,
       0,     0,     0,     0,     0,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   374,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    77,     0,     0,    78,     0,     0,    79,     0,
      80,   121,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,     0,     0,    75,    76,     0,
       0,     0,     0,     0,     0,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,    77,   417,     0,
      78,     0,   362,    79,     0,    80,     0,    61,    62,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,     0,     0,    75,    76,    77,     0,     0,    78,     0,
       0,    79,    19,    80,    20,   128,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,     0,
       0,    75,    76,     0,     0,     0,     0,     0,     0,     0,
      19,     0,    20,     0,     0,   121,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,     0,
       0,    75,    76,     0,     0,    77,     0,     0,    78,     0,
      19,    79,    20,    80,   121,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,    73,    74,     0,     0,
      75,    76,     0,     0,     0,     0,     0,     0,     0,    19,
       0,    20,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    77,     0,     0,    78,     0,     0,    79,     0,    80,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    77,
       0,     0,    78,     0,     0,    79,     0,    80,    75,    76,
       0,   300,   301,   302,     0,     0,     0,    19,     0,    20,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    77,
       0,     0,    78,     0,     0,    79,     0,    80,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    77,     0,
       0,    78,     0,     0,    79,     0,    80,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,   166,     0,     0,     0,     0,     0,
     247,     0,   248,   173,   174,   175,   176,     0,   249,   250,
     251,   181,   182,   183,   184,   185,   186,   187,   188,   189,
     190,   191,   192,   193,     0,     0,     0,     0,     0,   303,
       0,     0,   304,     0,   305,     0,     0,   306,   238,   239,
      75,    76,   240,   241,   242,   243,   244,   245,     0,    19,
       0,    20,   -54,    19,     0,    20,     0,     0,     0,     6,
     -54,   -54,     0,     0,     0,     0,     0,   -54,   -54,   -54,
     -54,   -54,   -54,   -54,     0,   -54,    21,     0,     0,     0,
       0,     0,   246,    22,     0,     0,     0,    23,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   149,
     150,   151,   152,   153,   154,   155,   156,   157,   158,   159,
     160,   161,   162,   163,   164,   165,   166,     0,     0,     0,
       0,     0,   247,     0,   248,   173,   174,   175,   176,     0,
     249,   250,   251,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,     0,     0,     0,     0,
       0,   238,   239,     0,   252,   240,   241,   242,   243,   244,
     245,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   246,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
       0,     0,     0,     0,     0,   247,     0,   248,   173,   174,
     175,   176,     0,   249,   250,   251,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   144,
       0,     0,     0,     0,     0,     0,     0,   252,     0,     0,
     145,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     146,   147,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   148,     0,     0,   149,
     150,   151,   152,   153,   154,   155,   156,   157,   158,   159,
     160,   161,   162,   163,   164,   165,   166,   167,   168,   169,
       0,     0,   170,   171,   172,   173,   174,   175,   176,   177,
     178,   179,   180,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193
};

static const short int yycheck[] =
{
      38,    98,    98,   131,   132,     3,   269,    51,    29,   148,
      32,    80,    33,    51,    10,   143,    96,   212,   213,   465,
     491,    37,   128,    52,    53,    54,   129,    10,   131,   132,
     132,   137,     4,   504,   229,   137,   126,    28,     0,    30,
     143,    57,    71,    15,   134,    43,   492,     5,     6,     7,
       8,     9,    10,    11,    12,   126,   126,   126,    96,    61,
     199,    57,   132,   134,    29,   145,     5,     6,     7,     8,
       9,    10,    11,    12,    57,   125,   114,   115,   116,    61,
     126,    72,    73,    74,    82,    76,    77,   126,   125,   169,
     136,   126,   172,   132,   303,   304,   305,   177,   178,   179,
     180,   136,   134,    30,    45,   137,    47,   145,    53,    54,
      55,    18,   117,   135,    28,   136,    30,   197,   198,    35,
      36,    55,    56,    99,   100,   228,    25,    26,   125,   167,
     168,   169,   170,   171,   172,    17,    18,   125,     0,   177,
     178,   179,   180,   271,    64,    65,    66,    67,    68,    69,
      70,   279,   280,   281,    17,    18,   194,   195,   196,   197,
     198,   230,   231,   232,    17,    18,   361,   130,   271,    31,
      17,    18,    34,   436,   132,   438,   279,   280,   281,   125,
      42,    43,    44,    45,    46,    47,    48,    17,    18,   125,
     128,    18,    30,   132,    18,    18,   234,   128,    18,    30,
      62,    27,    30,   131,   128,   131,   286,    59,   336,   126,
      18,    18,    18,    18,   342,    18,    42,    43,    44,    45,
      46,    47,    48,   267,    50,    21,   354,   355,    21,   267,
      21,   126,   129,   336,    63,   128,   128,   128,   128,   342,
     284,   128,   128,   128,   128,   128,   284,   285,   286,   318,
     128,   354,   355,   126,   126,   126,   126,   126,    39,   126,
     388,   389,   130,   391,   344,   393,   346,   347,   348,   397,
     126,   126,   352,    30,    30,   403,   404,   405,   126,   129,
     126,   126,   126,    30,    15,   388,   389,    15,   391,   126,
     393,   130,   126,   126,   397,    30,    18,    37,    39,    18,
     403,   404,   405,   128,   126,   126,   344,   345,   346,   347,
     348,   126,   126,   441,   352,   353,   314,   322,   323,   324,
     325,   326,   450,    57,   329,   330,   331,   332,   333,   126,
     128,   126,   126,    30,   372,    18,   126,   126,   441,   126,
     126,   126,   126,   126,   126,   473,   126,   450,     5,     6,
       7,     8,     9,    10,    11,    12,   394,   129,   126,   130,
     126,   489,   126,   443,   444,   445,   126,   129,   465,   465,
     473,   126,   410,   126,    15,   129,   381,   129,   129,   129,
     129,   129,    57,   129,   512,   513,   489,   132,    39,   517,
     132,   471,   520,   129,   432,   492,   492,   435,    15,   129,
     129,   126,    15,   126,    15,   443,   444,   445,    76,   512,
     513,   449,    15,   101,   517,   101,   101,   520,   423,   424,
     425,   426,   101,   428,   429,   430,   431,   101,    43,    24,
     373,   126,    50,   471,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    29,   228,
      19,    20,     3,   372,    41,   449,    -1,   394,   376,    28,
      -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   480,    -1,   482,   483,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    28,    -1,    30,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,    -1,
      -1,   130,    -1,    -1,   133,    -1,   135,   136,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   127,    -1,    40,   130,    -1,    -1,   133,
      -1,   135,   136,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    -1,    -1,    19,
      20,   127,    -1,    -1,   130,    -1,    -1,   133,    28,   135,
      30,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   127,    -1,    -1,   130,    -1,    -1,   133,    -1,
     135,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,    -1,    19,    20,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    28,    -1,    30,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,    40,    -1,
     130,    -1,   132,   133,    -1,   135,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    -1,    19,    20,   127,    -1,    -1,   130,    -1,
      -1,   133,    28,   135,    30,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
      -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      28,    -1,    30,    -1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
      -1,    19,    20,    -1,    -1,   127,    -1,    -1,   130,    -1,
      28,   133,    30,   135,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    -1,
      19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    28,
      -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   127,    -1,    -1,   130,    -1,    -1,   133,    -1,   135,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,
      -1,    -1,   130,    -1,    -1,   133,    -1,   135,    19,    20,
      -1,    22,    23,    24,    -1,    -1,    -1,    28,    -1,    30,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,
      -1,    -1,   130,    -1,    -1,   133,    -1,   135,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   127,    -1,
      -1,   130,    -1,    -1,   133,    -1,   135,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    -1,    -1,    -1,    -1,    -1,
     101,    -1,   103,   104,   105,   106,   107,    -1,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,    -1,    -1,    -1,    -1,    -1,   130,
      -1,    -1,   133,    -1,   135,    -1,    -1,   138,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    -1,    28,
      -1,    30,    27,    28,    -1,    30,    -1,    -1,    -1,    34,
      35,    36,    -1,    -1,    -1,    -1,    -1,    42,    43,    44,
      45,    46,    47,    48,    -1,    50,    51,    -1,    -1,    -1,
      -1,    -1,    61,    58,    -1,    -1,    -1,    62,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    -1,    -1,    -1,
      -1,    -1,   101,    -1,   103,   104,   105,   106,   107,    -1,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,    -1,    -1,    -1,    -1,
      -1,    17,    18,    -1,   133,    21,    22,    23,    24,    25,
      26,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      -1,    -1,    -1,    -1,    -1,   101,    -1,   103,   104,   105,
     106,   107,    -1,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,    38,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   133,    -1,    -1,
      49,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      59,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    75,    -1,    -1,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      -1,    -1,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   171,   172,   173,     0,    31,    34,    42,    43,    44,
      45,    46,    47,    48,    62,   152,   186,   188,   190,    28,
      30,    51,    58,    62,   151,   179,   190,    45,    47,   189,
      61,    64,    65,    66,    67,    68,    69,    70,   153,   184,
      29,   196,   197,   198,    52,    53,    54,    71,   176,   125,
      61,    27,    45,    47,    50,   152,   125,   184,    30,   174,
      18,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    19,    20,   127,   130,   133,
     135,   140,   160,   161,   162,   163,   164,   179,   193,    32,
     135,   185,    33,   136,   187,   197,    72,    73,    74,    76,
      77,   151,   199,   201,   125,   125,   125,   125,   130,   177,
     174,   160,    35,    36,   170,   170,   170,   170,    18,    18,
      18,     3,   136,   164,   165,   179,   128,   137,     3,   162,
     195,     4,    15,     5,     6,     7,     8,     9,    10,    11,
      12,   147,   148,   149,    38,    49,    59,    60,    75,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   142,   143,   144,   145,   146,   205,
     206,   208,   209,    30,    55,    56,   175,    18,    30,    30,
     178,   162,   162,   162,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,   147,   148,   150,   162,   167,
     131,   131,   126,   136,   128,    40,   165,   166,    17,    18,
      21,    22,    23,    24,    25,    26,    61,   101,   103,   109,
     110,   111,   133,   142,   143,   144,   145,   146,   168,   192,
     193,   194,   194,   194,   194,   195,    59,   153,   162,   162,
     195,   162,   162,   202,   195,   195,   195,   195,   195,   162,
     162,   162,   195,   195,   153,    99,   100,   126,   132,   126,
     158,   158,   158,    25,    26,    17,    18,   141,    18,    21,
      22,    23,    24,   130,   133,   135,   138,   168,   193,   158,
     164,   164,   164,    40,   162,   181,   182,   183,   126,   129,
      63,   191,   128,   128,   128,   128,   128,   167,   169,   128,
     128,   128,   128,   128,   126,   126,   160,   126,   155,   126,
     155,   194,   130,   126,   126,   126,   126,   126,   126,   194,
     194,   194,   126,    39,   160,   162,   195,    30,    37,    57,
     156,   159,   132,   169,   169,   136,   169,    30,   132,   134,
     179,   180,   126,   129,    40,    30,   167,   167,   167,   167,
     167,   126,   134,   167,   167,   167,   167,   167,    15,    15,
     194,    10,    57,    10,   126,   207,   194,   130,   195,   162,
     195,   195,   195,   126,   126,   126,   195,   162,   194,   194,
     126,    30,    18,   158,   132,   134,   136,    40,   181,   156,
     157,   126,   207,   126,   126,   126,   126,   167,   126,   126,
     126,   126,    39,   194,   194,   128,   194,    18,   194,   195,
     203,   126,   194,   126,   126,   126,   194,   194,   194,   128,
     162,    57,   154,    30,   129,   167,   167,   167,   167,   167,
     167,   167,   167,   162,   126,   130,   203,   204,   126,   155,
     155,   126,   194,   126,   195,   195,   195,   204,   194,    18,
     126,   129,   126,   126,   129,   129,   129,   129,   129,    15,
     132,   149,   200,   129,   195,   132,   194,   129,   167,   167,
     167,   194,   192,   132,   149,    39,   132,   129,   129,   129,
     126,   192,    15,    15,   126,   194,   194,    15,    76,   194,
      15,   194
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
        case 53:
#line 290 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 54:
#line 293 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string(""); 
  ;}
    break;

  case 62:
#line 300 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 69:
#line 305 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += *(yyvsp[0].String); 
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
    ;}
    break;

  case 70:
#line 310 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 71:
#line 315 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 72:
#line 316 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { *(yyvsp[-1].String) += " " + *(yyvsp[0].String); delete (yyvsp[0].String); (yyval.String) = (yyvsp[-1].String); ;}
    break;

  case 73:
#line 319 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 74:
#line 320 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, ", "); 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 75:
#line 328 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 76:
#line 334 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 78:
#line 338 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 79:
#line 339 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      (yyvsp[-1].String)->insert(0, ", ");
      if (!(yyvsp[0].String)->empty())
        *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 81:
#line 349 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 99:
#line 371 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type).newTy = (yyvsp[0].String); 
    (yyval.Type).oldTy = OpaqueTy; 
  ;}
    break;

  case 100:
#line 375 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type).newTy = (yyvsp[0].String);
    (yyval.Type).oldTy = UnresolvedTy;
  ;}
    break;

  case 102:
#line 384 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Type UpReference
    (yyvsp[0].String)->insert(0, "\\");
    (yyval.Type).newTy = (yyvsp[0].String);
    (yyval.Type).oldTy = NumericTy;
  ;}
    break;

  case 103:
#line 389 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {           // Function derived type?
    *(yyvsp[-3].Type).newTy += "( " + *(yyvsp[-1].String) + " )";
    delete (yyvsp[-1].String);
    (yyval.Type).newTy = (yyvsp[-3].Type).newTy;
    (yyval.Type).oldTy = FunctionTy;
  ;}
    break;

  case 104:
#line 395 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Sized array type?
    (yyvsp[-3].String)->insert(0,"[ ");
    *(yyvsp[-3].String) += " x " + *(yyvsp[-1].Type).newTy + " ]";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].String);
    (yyval.Type).oldTy = ArrayTy;
  ;}
    break;

  case 105:
#line 402 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Packed array type?
    (yyvsp[-3].String)->insert(0,"< ");
    *(yyvsp[-3].String) += " x " + *(yyvsp[-1].Type).newTy + " >";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].String);
    (yyval.Type).oldTy = PackedTy;
  ;}
    break;

  case 106:
#line 409 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                        // Structure type?
    (yyvsp[-1].String)->insert(0, "{ ");
    *(yyvsp[-1].String) += " }";
    (yyval.Type).newTy = (yyvsp[-1].String);
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 107:
#line 415 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                  // Empty structure type?
    (yyval.Type).newTy = new std::string("{}");
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 108:
#line 419 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                             // Pointer type?
    *(yyvsp[-1].Type).newTy += '*';
    (yyvsp[-1].Type).oldTy = PointerTy;
    (yyval.Type) = (yyvsp[-1].Type);
  ;}
    break;

  case 109:
#line 429 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].Type).newTy;
  ;}
    break;

  case 110:
#line 432 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Type).newTy;
    delete (yyvsp[0].Type).newTy;
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 112:
#line 441 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 113:
#line 446 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 114:
#line 449 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 115:
#line 459 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " [ " + *(yyvsp[-1].String) + " ]";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 116:
#line 465 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += "[ ]";
  ;}
    break;

  case 117:
#line 470 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " c" + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 118:
#line 476 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " < " + *(yyvsp[-1].String) + " >";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 119:
#line 482 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " { " + *(yyvsp[-1].String) + " }";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 120:
#line 488 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " {}";
  ;}
    break;

  case 121:
#line 493 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst +=  " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 122:
#line 499 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 123:
#line 505 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 124:
#line 511 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 125:
#line 517 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 126:
#line 523 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {      // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 127:
#line 529 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {            // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 128:
#line 535 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                      // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 129:
#line 541 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                     // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 130:
#line 547 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Float & Double constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 131:
#line 555 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string source = *(yyvsp[-3].Const).cnst;
    TypeInfo DstTy = (yyvsp[-1].Type);
    ResolveType(DstTy);
    if (*(yyvsp[-5].String) == "cast") {
      // Call getCastUpgrade to upgrade the old cast
      (yyval.String) = new std::string(getCastUpgrade(source, (yyvsp[-3].Const).type, (yyvsp[-1].Type), true));
    } else {
      // Nothing to upgrade, just create the cast constant expr
      (yyval.String) = new std::string(*(yyvsp[-5].String));
      *(yyval.String) += "( " + source + " to " + *(yyvsp[-1].Type).newTy + ")";
    }
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy();
  ;}
    break;

  case 132:
#line 569 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += "(" + *(yyvsp[-2].Const).cnst;
    for (unsigned i = 0; i < (yyvsp[-1].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-1].ValList))[i];
      *(yyvsp[-4].String) += ", " + *VI.val;
      VI.destroy();
    }
    *(yyvsp[-4].String) += ")";
    (yyval.String) = (yyvsp[-4].String);
    (yyvsp[-2].Const).destroy();
    delete (yyvsp[-1].ValList);
  ;}
    break;

  case 133:
#line 581 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 134:
#line 586 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 135:
#line 591 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 136:
#line 596 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 137:
#line 601 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-5].String)->c_str();
    if (*(yyvsp[-5].String) == "shr")
      shiftop = ((yyvsp[-3].Const).type.isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
  ;}
    break;

  case 138:
#line 609 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 139:
#line 614 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 140:
#line 619 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 141:
#line 629 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 142:
#line 634 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Const).cnst); (yyvsp[0].Const).destroy(); ;}
    break;

  case 145:
#line 649 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
;}
    break;

  case 146:
#line 654 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 147:
#line 657 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 148:
#line 662 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "module asm " << " " << *(yyvsp[0].String) << "\n";
    (yyval.String) = 0;
  ;}
    break;

  case 149:
#line 666 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "implementation\n";
    (yyval.String) = 0;
  ;}
    break;

  case 150:
#line 670 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = 0; ;}
    break;

  case 151:
#line 673 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    EnumeratedTypes.push_back((yyvsp[0].Type));
    if (!(yyvsp[-2].String)->empty()) {
      NamedTypes[*(yyvsp[-2].String)].newTy = new std::string(*(yyvsp[0].Type).newTy);
      NamedTypes[*(yyvsp[-2].String)].oldTy = (yyvsp[0].Type).oldTy;
      *O << *(yyvsp[-2].String) << " = ";
    }
    *O << "type " << *(yyvsp[0].Type).newTy << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 152:
#line 684 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {       // Function prototypes can be in const pool
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 153:
#line 689 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  // Asm blocks can be in the const pool
    *O << *(yyvsp[-2].String) << " " << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 154:
#line 694 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Const).cnst << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Const).destroy(); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 155:
#line 701 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O <<  *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 156:
#line 708 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 157:
#line 715 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 158:
#line 722 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 159:
#line 727 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " = " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 160:
#line 732 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = 0;
  ;}
    break;

  case 164:
#line 742 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 165:
#line 747 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    if (*(yyvsp[0].String) == "64")
      SizeOfPointer = 64;
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 166:
#line 754 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 167:
#line 759 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 168:
#line 766 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1].String)->insert(0, "[ ");
    *(yyvsp[-1].String) += " ]";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 169:
#line 773 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 171:
#line 779 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 175:
#line 788 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 176:
#line 790 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  (yyval.String) = (yyvsp[-1].Type).newTy;
  if (!(yyvsp[0].String)->empty())
    *(yyval.String) += " " + *(yyvsp[0].String);
  delete (yyvsp[0].String);
;}
    break;

  case 177:
#line 797 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 178:
#line 801 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 179:
#line 805 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 180:
#line 808 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    (yyval.String) = (yyvsp[-2].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 181:
#line 813 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 182:
#line 816 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 183:
#line 819 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 184:
#line 838 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string("begin");
  ;}
    break;

  case 185:
#line 841 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = new std::string ("{");
  ;}
    break;

  case 186:
#line 845 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if (!(yyvsp[-2].String)->empty()) {
    *O << *(yyvsp[-2].String) << " ";
  }
  *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
  delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String);
  (yyval.String) = 0;
;}
    break;

  case 187:
#line 854 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("end"); ;}
    break;

  case 188:
#line 855 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); ;}
    break;

  case 189:
#line 857 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if ((yyvsp[-1].String))
    *O << *(yyvsp[-1].String);
  *O << '\n' << *(yyvsp[0].String) << "\n";
  (yyval.String) = 0;
;}
    break;

  case 190:
#line 865 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 193:
#line 871 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[-1].String);
    *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 194:
#line 884 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 204:
#line 890 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, "<");
    *(yyvsp[-1].String) += ">";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 206:
#line 896 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty()) {
      *(yyvsp[-4].String) += " " + *(yyvsp[-3].String);
    }
    *(yyvsp[-4].String) += " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 211:
#line 914 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).type = (yyvsp[-1].Type);
    (yyval.Value).val = new std::string(*(yyvsp[-1].Type).newTy + " ");
    *(yyval.Value).val += *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 212:
#line 921 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 213:
#line 924 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Do not allow functions with 0 basic blocks   
    (yyval.String) = 0;
  ;}
    break;

  case 214:
#line 932 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 215:
#line 936 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 216:
#line 941 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 217:
#line 944 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 218:
#line 950 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {              // Return with a result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 219:
#line 955 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                       // Return with no result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Type).newTy << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 220:
#line 960 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                         // Unconditional Branch...
    *O << "    " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 221:
#line 965 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].String) << ", " 
       << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].String) << ", " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); delete (yyvsp[-6].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); 
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 222:
#line 972 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].String) << ", " << *(yyvsp[-4].Type).newTy 
       << " " << *(yyvsp[-3].String) << " [" << *(yyvsp[-1].String) << " ]\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); delete (yyvsp[-6].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 223:
#line 978 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-7].String) << " " << *(yyvsp[-6].Type).newTy << " " << *(yyvsp[-5].String) << ", " 
       << *(yyvsp[-3].Type).newTy << " " << *(yyvsp[-2].String) << "[]\n";
    delete (yyvsp[-7].String); (yyvsp[-6].Type).destroy(); delete (yyvsp[-5].String); (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String);
    (yyval.String) = 0;
  ;}
    break;

  case 224:
#line 985 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    ";
    if (!(yyvsp[-13].String)->empty())
      *O << *(yyvsp[-13].String) << " = ";
    *O << *(yyvsp[-12].String) << " " << *(yyvsp[-11].String) << " " << *(yyvsp[-10].Type).newTy << " " << *(yyvsp[-9].String) << " (";
    for (unsigned i = 0; i < (yyvsp[-7].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-7].ValList))[i];
      *O << *VI.val;
      if (i+1 < (yyvsp[-7].ValList)->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *(yyvsp[-5].String) << " " << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].String) << " " 
       << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-13].String); delete (yyvsp[-12].String); delete (yyvsp[-11].String); (yyvsp[-10].Type).destroy(); delete (yyvsp[-9].String); delete (yyvsp[-7].ValList); 
    delete (yyvsp[-5].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); 
    delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 225:
#line 1004 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 226:
#line 1009 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 227:
#line 1015 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].String) + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 228:
#line 1020 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-3].String)->insert(0, *(yyvsp[-4].Type).newTy + " " );
    *(yyvsp[-3].String) += ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 229:
#line 1028 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-1].String) += " = ";
    *(yyvsp[-1].String) += *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
  ;}
    break;

  case 230:
#line 1037 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {    // Used for PHI nodes
    (yyvsp[-3].String)->insert(0, *(yyvsp[-5].Type).newTy + "[");
    *(yyvsp[-3].String) += "," + *(yyvsp[-1].String) + "]";
    (yyvsp[-5].Type).destroy(); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 231:
#line 1043 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6].String) += ", [" + *(yyvsp[-3].String) + "," + *(yyvsp[-1].String) + "]";
    delete (yyvsp[-3].String); delete (yyvsp[-1].String);
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 232:
#line 1051 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.ValList) = new ValueList();
    (yyval.ValList)->push_back((yyvsp[0].Value));
  ;}
    break;

  case 233:
#line 1055 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-2].ValList)->push_back((yyvsp[0].Value));
    (yyval.ValList) = (yyvsp[-2].ValList);
  ;}
    break;

  case 234:
#line 1062 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = (yyvsp[0].ValList); ;}
    break;

  case 235:
#line 1063 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = new ValueList(); ;}
    break;

  case 236:
#line 1067 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 238:
#line 1075 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 239:
#line 1080 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 240:
#line 1085 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    (yyvsp[-3].Type).destroy(); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 241:
#line 1090 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 242:
#line 1095 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-3].String)->c_str();
    if (*(yyvsp[-3].String) == "shr")
      shiftop = ((yyvsp[-2].Value).type.isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 243:
#line 1103 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string source = *(yyvsp[-2].Value).val;
    TypeInfo SrcTy = (yyvsp[-2].Value).type;
    TypeInfo DstTy = (yyvsp[0].Type);
    ResolveType(DstTy);
    (yyval.String) = new std::string();
    if (*(yyvsp[-3].String) == "cast") {
      *(yyval.String) +=  getCastUpgrade(source, SrcTy, DstTy, false);
    } else {
      *(yyval.String) += *(yyvsp[-3].String) + " " + source + " to " + *DstTy.newTy;
    }
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy();
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
  ;}
    break;

  case 244:
#line 1117 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 245:
#line 1122 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Type).newTy;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Type).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 246:
#line 1127 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 247:
#line 1132 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 248:
#line 1137 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 249:
#line 1142 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 250:
#line 1147 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-6].String) += " " + *(yyvsp[-5].String);
    if (!(yyvsp[-6].String)->empty())
      *(yyvsp[-6].String) += " ";
    *(yyvsp[-6].String) += *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].String) + "(";
    for (unsigned i = 0; i < (yyvsp[-1].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-1].ValList))[i];
      *(yyvsp[-6].String) += *VI.val;
      if (i+1 < (yyvsp[-1].ValList)->size())
        *(yyvsp[-6].String) += ", ";
      VI.destroy();
    }
    *(yyvsp[-6].String) += ")";
    delete (yyvsp[-5].String); (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); delete (yyvsp[-1].ValList);
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 252:
#line 1169 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = (yyvsp[0].ValList); ;}
    break;

  case 253:
#line 1170 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  (yyval.ValList) = new ValueList(); ;}
    break;

  case 255:
#line 1175 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 256:
#line 1178 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 257:
#line 1185 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 258:
#line 1192 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 259:
#line 1199 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 260:
#line 1206 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 261:
#line 1211 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty())
      *(yyvsp[-3].String) += " ";
    *(yyvsp[-3].String) += *(yyvsp[-2].String) + " " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 262:
#line 1218 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-5].String) += " ";
    *(yyvsp[-5].String) += *(yyvsp[-4].String) + " " + *(yyvsp[-3].Value).val + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].String);
    delete (yyvsp[-4].String); (yyvsp[-3].Value).destroy(); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 263:
#line 1225 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].String);
    for (unsigned i = 0; i < (yyvsp[0].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[0].ValList))[i];
      *(yyvsp[-3].String) += ", " + *VI.val;
      VI.destroy();
    }
    (yyvsp[-2].Type).destroy(); delete (yyvsp[-1].String); delete (yyvsp[0].ValList);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 3571 "UpgradeParser.tab.c"

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


#line 1236 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"


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

