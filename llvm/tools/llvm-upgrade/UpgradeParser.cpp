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
     UNINITIALIZED = 313,
     DEPLIBS = 314,
     CALL = 315,
     TAIL = 316,
     ASM_TOK = 317,
     MODULE = 318,
     SIDEEFFECT = 319,
     CC_TOK = 320,
     CCC_TOK = 321,
     CSRETCC_TOK = 322,
     FASTCC_TOK = 323,
     COLDCC_TOK = 324,
     X86_STDCALLCC_TOK = 325,
     X86_FASTCALLCC_TOK = 326,
     DATALAYOUT = 327,
     RET = 328,
     BR = 329,
     SWITCH = 330,
     INVOKE = 331,
     EXCEPT = 332,
     UNWIND = 333,
     UNREACHABLE = 334,
     ADD = 335,
     SUB = 336,
     MUL = 337,
     DIV = 338,
     UDIV = 339,
     SDIV = 340,
     FDIV = 341,
     REM = 342,
     UREM = 343,
     SREM = 344,
     FREM = 345,
     AND = 346,
     OR = 347,
     XOR = 348,
     SETLE = 349,
     SETGE = 350,
     SETLT = 351,
     SETGT = 352,
     SETEQ = 353,
     SETNE = 354,
     MALLOC = 355,
     ALLOCA = 356,
     FREE = 357,
     LOAD = 358,
     STORE = 359,
     GETELEMENTPTR = 360,
     PHI_TOK = 361,
     SELECT = 362,
     SHL = 363,
     SHR = 364,
     ASHR = 365,
     LSHR = 366,
     VAARG = 367,
     EXTRACTELEMENT = 368,
     INSERTELEMENT = 369,
     SHUFFLEVECTOR = 370,
     CAST = 371,
     TRUNC = 372,
     ZEXT = 373,
     SEXT = 374,
     FPTRUNC = 375,
     FPEXT = 376,
     FPTOUI = 377,
     FPTOSI = 378,
     UITOFP = 379,
     SITOFP = 380,
     PTRTOINT = 381,
     INTTOPTR = 382,
     BITCAST = 383
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
#define UNINITIALIZED 313
#define DEPLIBS 314
#define CALL 315
#define TAIL 316
#define ASM_TOK 317
#define MODULE 318
#define SIDEEFFECT 319
#define CC_TOK 320
#define CCC_TOK 321
#define CSRETCC_TOK 322
#define FASTCC_TOK 323
#define COLDCC_TOK 324
#define X86_STDCALLCC_TOK 325
#define X86_FASTCALLCC_TOK 326
#define DATALAYOUT 327
#define RET 328
#define BR 329
#define SWITCH 330
#define INVOKE 331
#define EXCEPT 332
#define UNWIND 333
#define UNREACHABLE 334
#define ADD 335
#define SUB 336
#define MUL 337
#define DIV 338
#define UDIV 339
#define SDIV 340
#define FDIV 341
#define REM 342
#define UREM 343
#define SREM 344
#define FREM 345
#define AND 346
#define OR 347
#define XOR 348
#define SETLE 349
#define SETGE 350
#define SETLT 351
#define SETGT 352
#define SETEQ 353
#define SETNE 354
#define MALLOC 355
#define ALLOCA 356
#define FREE 357
#define LOAD 358
#define STORE 359
#define GETELEMENTPTR 360
#define PHI_TOK 361
#define SELECT 362
#define SHL 363
#define SHR 364
#define ASHR 365
#define LSHR 366
#define VAARG 367
#define EXTRACTELEMENT 368
#define INSERTELEMENT 369
#define SHUFFLEVECTOR 370
#define CAST 371
#define TRUNC 372
#define ZEXT 373
#define SEXT 374
#define FPTRUNC 375
#define FPEXT 376
#define FPTOUI 377
#define FPTOSI 378
#define UITOFP 379
#define SITOFP 380
#define PTRTOINT 381
#define INTTOPTR 382
#define BITCAST 383




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
static uint64_t unique = 1;

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
#line 239 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
typedef union YYSTYPE {
  std::string*    String;
  TypeInfo        Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
} YYSTYPE;
/* Line 196 of yacc.c.  */
#line 580 "UpgradeParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 592 "UpgradeParser.tab.c"

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
#define YYLAST   1324

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  143
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  73
/* YYNRULES -- Number of rules. */
#define YYNRULES  269
/* YYNRULES -- Number of states. */
#define YYNSTATES  528

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   383

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     132,   133,   141,     2,   130,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     137,   129,   138,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   134,   131,   136,     2,     2,     2,     2,     2,   142,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     135,     2,     2,   139,     2,   140,     2,     2,     2,     2,
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
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128
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
      99,   101,   103,   105,   107,   109,   112,   113,   115,   117,
     119,   121,   123,   125,   127,   128,   130,   132,   134,   136,
     138,   140,   143,   144,   145,   148,   149,   153,   156,   157,
     159,   160,   164,   166,   169,   171,   173,   175,   177,   179,
     181,   183,   185,   187,   189,   191,   193,   195,   197,   199,
     201,   203,   205,   207,   209,   212,   217,   223,   229,   233,
     236,   239,   241,   245,   247,   251,   253,   254,   259,   263,
     267,   272,   277,   281,   284,   287,   290,   293,   296,   299,
     302,   305,   308,   311,   318,   324,   333,   340,   347,   354,
     361,   368,   377,   386,   390,   392,   394,   396,   398,   401,
     404,   409,   412,   414,   416,   418,   423,   426,   431,   438,
     445,   452,   459,   463,   468,   469,   471,   473,   475,   479,
     483,   487,   491,   495,   499,   501,   502,   504,   506,   508,
     509,   512,   516,   518,   520,   524,   526,   527,   536,   538,
     540,   544,   546,   548,   552,   553,   555,   557,   561,   562,
     564,   566,   568,   570,   572,   574,   576,   578,   580,   584,
     586,   592,   594,   596,   598,   600,   603,   606,   608,   611,
     614,   615,   617,   619,   621,   624,   627,   631,   641,   651,
     660,   675,   677,   679,   686,   692,   695,   702,   710,   712,
     716,   718,   719,   722,   724,   730,   736,   742,   745,   750,
     755,   762,   767,   772,   779,   786,   789,   797,   799,   802,
     803,   805,   806,   810,   817,   821,   828,   831,   836,   843
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     175,     0,    -1,    19,    -1,    20,    -1,    17,    -1,    18,
      -1,    80,    -1,    81,    -1,    82,    -1,    83,    -1,    84,
      -1,    85,    -1,    86,    -1,    87,    -1,    88,    -1,    89,
      -1,    90,    -1,    91,    -1,    92,    -1,    93,    -1,    94,
      -1,    95,    -1,    96,    -1,    97,    -1,    98,    -1,    99,
      -1,   108,    -1,   109,    -1,   110,    -1,   111,    -1,   117,
      -1,   118,    -1,   119,    -1,   120,    -1,   121,    -1,   122,
      -1,   123,    -1,   124,    -1,   125,    -1,   126,    -1,   127,
      -1,   128,    -1,   116,    -1,    11,    -1,     9,    -1,     7,
      -1,     5,    -1,    12,    -1,    10,    -1,     8,    -1,     6,
      -1,   151,    -1,   152,    -1,    13,    -1,    14,    -1,   184,
     129,    -1,    -1,    42,    -1,    43,    -1,    44,    -1,    48,
      -1,    45,    -1,    46,    -1,    47,    -1,    -1,    66,    -1,
      67,    -1,    68,    -1,    69,    -1,    70,    -1,    71,    -1,
      65,    18,    -1,    -1,    -1,    57,    18,    -1,    -1,   130,
      57,    18,    -1,    37,    30,    -1,    -1,   160,    -1,    -1,
     130,   163,   162,    -1,   160,    -1,    57,    18,    -1,   166,
      -1,     3,    -1,   168,    -1,     3,    -1,   168,    -1,     4,
      -1,     5,    -1,     6,    -1,     7,    -1,     8,    -1,     9,
      -1,    10,    -1,    11,    -1,    12,    -1,    13,    -1,    14,
      -1,    15,    -1,    16,    -1,   198,    -1,   167,    -1,   131,
      18,    -1,   165,   132,   170,   133,    -1,   134,    18,   135,
     168,   136,    -1,   137,    18,   135,   168,   138,    -1,   139,
     169,   140,    -1,   139,   140,    -1,   168,   141,    -1,   168,
      -1,   169,   130,   168,    -1,   169,    -1,   169,   130,    40,
      -1,    40,    -1,    -1,   166,   134,   173,   136,    -1,   166,
     134,   136,    -1,   166,   142,    30,    -1,   166,   137,   173,
     138,    -1,   166,   139,   173,   140,    -1,   166,   139,   140,
      -1,   166,    22,    -1,   166,    23,    -1,   166,   198,    -1,
     166,   172,    -1,   166,    24,    -1,   151,   145,    -1,   152,
      18,    -1,     4,    25,    -1,     4,    26,    -1,   154,    21,
      -1,   150,   132,   171,    39,   166,   133,    -1,   105,   132,
     171,   213,   133,    -1,   107,   132,   171,   130,   171,   130,
     171,   133,    -1,   146,   132,   171,   130,   171,   133,    -1,
     147,   132,   171,   130,   171,   133,    -1,   148,   132,   171,
     130,   171,   133,    -1,   149,   132,   171,   130,   171,   133,
      -1,   113,   132,   171,   130,   171,   133,    -1,   114,   132,
     171,   130,   171,   130,   171,   133,    -1,   115,   132,   171,
     130,   171,   130,   171,   133,    -1,   173,   130,   171,    -1,
     171,    -1,    35,    -1,    36,    -1,   176,    -1,   176,   193,
      -1,   176,   195,    -1,   176,    63,    62,   179,    -1,   176,
      31,    -1,   178,    -1,    50,    -1,    58,    -1,   178,   155,
      27,   164,    -1,   178,   195,    -1,   178,    63,    62,   179,
      -1,   178,   155,   156,   174,   171,   162,    -1,   178,   155,
     177,   174,   166,   162,    -1,   178,   155,    45,   174,   166,
     162,    -1,   178,   155,    47,   174,   166,   162,    -1,   178,
      51,   181,    -1,   178,    59,   129,   182,    -1,    -1,    30,
      -1,    56,    -1,    55,    -1,    53,   129,   180,    -1,    54,
     129,    18,    -1,    52,   129,    30,    -1,    72,   129,    30,
      -1,   134,   183,   136,    -1,   183,   130,    30,    -1,    30,
      -1,    -1,    28,    -1,    30,    -1,   184,    -1,    -1,   166,
     185,    -1,   187,   130,   186,    -1,   186,    -1,   187,    -1,
     187,   130,    40,    -1,    40,    -1,    -1,   157,   164,   184,
     132,   188,   133,   161,   158,    -1,    32,    -1,   139,    -1,
     156,   189,   190,    -1,    33,    -1,   140,    -1,   191,   201,
     192,    -1,    -1,    45,    -1,    47,    -1,    34,   194,   189,
      -1,    -1,    64,    -1,    17,    -1,    18,    -1,    21,    -1,
      25,    -1,    26,    -1,    22,    -1,    23,    -1,    24,    -1,
     137,   173,   138,    -1,   172,    -1,    62,   196,    30,   130,
      30,    -1,   144,    -1,   184,    -1,   198,    -1,   197,    -1,
     166,   199,    -1,   201,   202,    -1,   202,    -1,   203,   205,
      -1,   203,   207,    -1,    -1,    29,    -1,    78,    -1,    77,
      -1,    73,   200,    -1,    73,     3,    -1,    74,    15,   199,
      -1,    74,     4,   199,   130,    15,   199,   130,    15,   199,
      -1,    75,   153,   199,   130,    15,   199,   134,   206,   136,
      -1,    75,   153,   199,   130,    15,   199,   134,   136,    -1,
     155,    76,   157,   164,   199,   132,   210,   133,    39,    15,
     199,   204,    15,   199,    -1,   204,    -1,    79,    -1,   206,
     153,   197,   130,    15,   199,    -1,   153,   197,   130,    15,
     199,    -1,   155,   212,    -1,   166,   134,   199,   130,   199,
     136,    -1,   208,   130,   134,   199,   130,   199,   136,    -1,
     200,    -1,   209,   130,   200,    -1,   209,    -1,    -1,    61,
      60,    -1,    60,    -1,   146,   166,   199,   130,   199,    -1,
     147,   166,   199,   130,   199,    -1,   148,   166,   199,   130,
     199,    -1,    49,   200,    -1,   149,   200,   130,   200,    -1,
     150,   200,    39,   166,    -1,   107,   200,   130,   200,   130,
     200,    -1,   112,   200,   130,   166,    -1,   113,   200,   130,
     200,    -1,   114,   200,   130,   200,   130,   200,    -1,   115,
     200,   130,   200,   130,   200,    -1,   106,   208,    -1,   211,
     157,   164,   199,   132,   210,   133,    -1,   215,    -1,   130,
     209,    -1,    -1,    38,    -1,    -1,   100,   166,   159,    -1,
     100,   166,   130,    10,   199,   159,    -1,   101,   166,   159,
      -1,   101,   166,   130,    10,   199,   159,    -1,   102,   200,
      -1,   214,   103,   166,   199,    -1,   214,   104,   200,   130,
     166,   199,    -1,   105,   166,   199,   213,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   299,   299,   299,   300,   300,   304,   304,   304,   304,
     304,   304,   304,   305,   305,   305,   305,   306,   306,   306,
     307,   307,   307,   307,   307,   307,   308,   308,   308,   308,
     309,   309,   309,   309,   309,   309,   309,   310,   310,   310,
     310,   310,   310,   315,   315,   315,   315,   316,   316,   316,
     316,   317,   317,   318,   318,   321,   324,   329,   329,   329,
     329,   329,   329,   330,   331,   334,   334,   334,   334,   334,
     335,   336,   341,   346,   347,   350,   351,   359,   365,   366,
     369,   370,   379,   380,   393,   393,   394,   394,   395,   399,
     399,   399,   399,   399,   399,   399,   400,   400,   400,   400,
     400,   402,   406,   410,   413,   418,   424,   432,   440,   446,
     450,   461,   464,   472,   473,   478,   481,   491,   497,   502,
     508,   514,   520,   525,   531,   537,   543,   549,   555,   561,
     567,   573,   579,   587,   601,   613,   618,   624,   629,   634,
     642,   647,   652,   662,   667,   672,   672,   682,   687,   690,
     695,   699,   703,   705,   705,   708,   720,   725,   730,   737,
     744,   751,   758,   763,   768,   773,   775,   775,   778,   783,
     790,   795,   802,   809,   814,   815,   823,   823,   824,   824,
     826,   833,   837,   841,   844,   849,   852,   854,   874,   875,
     877,   886,   887,   889,   897,   898,   899,   903,   916,   917,
     920,   920,   920,   920,   920,   920,   920,   921,   922,   927,
     928,   937,   937,   941,   947,   958,   964,   967,   975,   979,
     984,   987,   993,   993,   995,  1000,  1005,  1010,  1018,  1025,
    1031,  1051,  1056,  1062,  1067,  1075,  1084,  1091,  1099,  1103,
    1110,  1111,  1115,  1120,  1123,  1129,  1134,  1139,  1144,  1152,
    1166,  1171,  1176,  1181,  1186,  1191,  1196,  1213,  1218,  1219,
    1223,  1224,  1227,  1234,  1241,  1248,  1255,  1260,  1267,  1274
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
  "UNINITIALIZED", "DEPLIBS", "CALL", "TAIL", "ASM_TOK", "MODULE",
  "SIDEEFFECT", "CC_TOK", "CCC_TOK", "CSRETCC_TOK", "FASTCC_TOK",
  "COLDCC_TOK", "X86_STDCALLCC_TOK", "X86_FASTCALLCC_TOK", "DATALAYOUT",
  "RET", "BR", "SWITCH", "INVOKE", "EXCEPT", "UNWIND", "UNREACHABLE",
  "ADD", "SUB", "MUL", "DIV", "UDIV", "SDIV", "FDIV", "REM", "UREM",
  "SREM", "FREM", "AND", "OR", "XOR", "SETLE", "SETGE", "SETLT", "SETGT",
  "SETEQ", "SETNE", "MALLOC", "ALLOCA", "FREE", "LOAD", "STORE",
  "GETELEMENTPTR", "PHI_TOK", "SELECT", "SHL", "SHR", "ASHR", "LSHR",
  "VAARG", "EXTRACTELEMENT", "INSERTELEMENT", "SHUFFLEVECTOR", "CAST",
  "TRUNC", "ZEXT", "SEXT", "FPTRUNC", "FPEXT", "FPTOUI", "FPTOSI",
  "UITOFP", "SITOFP", "PTRTOINT", "INTTOPTR", "BITCAST", "'='", "','",
  "'\\\\'", "'('", "')'", "'['", "'x'", "']'", "'<'", "'>'", "'{'", "'}'",
  "'*'", "'c'", "$accept", "IntVal", "EInt64Val", "ArithmeticOps",
  "LogicalOps", "SetCondOps", "ShiftOps", "CastOps", "SIntType",
  "UIntType", "IntType", "FPType", "OptAssign", "OptLinkage",
  "OptCallingConv", "OptAlign", "OptCAlign", "SectionString", "OptSection",
  "GlobalVarAttributes", "GlobalVarAttribute", "TypesV", "UpRTypesV",
  "Types", "PrimType", "UpRTypes", "TypeListI", "ArgTypeListI", "ConstVal",
  "ConstExpr", "ConstVector", "GlobalType", "Module", "DefinitionList",
  "External", "ConstPool", "AsmBlock", "BigOrLittle", "TargetDefinition",
  "LibrariesDefinition", "LibList", "Name", "OptName", "ArgVal",
  "ArgListH", "ArgList", "FunctionHeaderH", "BEGIN", "FunctionHeader",
  "END", "Function", "FnDeclareLinkage", "FunctionProto", "OptSideEffect",
  "ConstValueRef", "SymbolicValueRef", "ValueRef", "ResolvedVal",
  "BasicBlockList", "BasicBlock", "InstructionList", "Unwind",
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
     375,   376,   377,   378,   379,   380,   381,   382,   383,    61,
      44,    92,    40,    41,    91,   120,    93,    60,    62,   123,
     125,    42,    99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   143,   144,   144,   145,   145,   146,   146,   146,   146,
     146,   146,   146,   146,   146,   146,   146,   147,   147,   147,
     148,   148,   148,   148,   148,   148,   149,   149,   149,   149,
     150,   150,   150,   150,   150,   150,   150,   150,   150,   150,
     150,   150,   150,   151,   151,   151,   151,   152,   152,   152,
     152,   153,   153,   154,   154,   155,   155,   156,   156,   156,
     156,   156,   156,   156,   156,   157,   157,   157,   157,   157,
     157,   157,   157,   158,   158,   159,   159,   160,   161,   161,
     162,   162,   163,   163,   164,   164,   165,   165,   166,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     168,   169,   169,   170,   170,   170,   170,   171,   171,   171,
     171,   171,   171,   171,   171,   171,   171,   171,   171,   171,
     171,   171,   171,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   173,   173,   174,   174,   175,   176,   176,
     176,   176,   176,   177,   177,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   179,   180,   180,   181,   181,
     181,   181,   182,   183,   183,   183,   184,   184,   185,   185,
     186,   187,   187,   188,   188,   188,   188,   189,   190,   190,
     191,   192,   192,   193,   194,   194,   194,   195,   196,   196,
     197,   197,   197,   197,   197,   197,   197,   197,   197,   197,
     197,   198,   198,   199,   199,   200,   201,   201,   202,   203,
     203,   203,   204,   204,   205,   205,   205,   205,   205,   205,
     205,   205,   205,   206,   206,   207,   208,   208,   209,   209,
     210,   210,   211,   211,   212,   212,   212,   212,   212,   212,
     212,   212,   212,   212,   212,   212,   212,   212,   213,   213,
     214,   214,   215,   215,   215,   215,   215,   215,   215,   215
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     2,     0,     1,     1,     1,
       1,     1,     1,     1,     0,     1,     1,     1,     1,     1,
       1,     2,     0,     0,     2,     0,     3,     2,     0,     1,
       0,     3,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     4,     5,     5,     3,     2,
       2,     1,     3,     1,     3,     1,     0,     4,     3,     3,
       4,     4,     3,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     6,     5,     8,     6,     6,     6,     6,
       6,     8,     8,     3,     1,     1,     1,     1,     2,     2,
       4,     2,     1,     1,     1,     4,     2,     4,     6,     6,
       6,     6,     3,     4,     0,     1,     1,     1,     3,     3,
       3,     3,     3,     3,     1,     0,     1,     1,     1,     0,
       2,     3,     1,     1,     3,     1,     0,     8,     1,     1,
       3,     1,     1,     3,     0,     1,     1,     3,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     1,
       5,     1,     1,     1,     1,     2,     2,     1,     2,     2,
       0,     1,     1,     1,     2,     2,     3,     9,     9,     8,
      14,     1,     1,     6,     5,     2,     6,     7,     1,     3,
       1,     0,     2,     1,     5,     5,     5,     2,     4,     4,
       6,     4,     4,     6,     6,     2,     7,     1,     2,     0,
       1,     0,     3,     6,     3,     6,     2,     4,     6,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned short int yydefact[] =
{
     164,     0,    64,   152,     1,   151,   194,    57,    58,    59,
      61,    62,    63,    60,     0,    72,   220,   148,   149,   176,
     177,     0,     0,     0,    64,     0,   156,   195,   196,    72,
       0,     0,    65,    66,    67,    68,    69,    70,     0,     0,
     221,   220,   217,    56,     0,     0,     0,     0,   162,     0,
       0,     0,     0,     0,   153,   154,     0,     0,    55,   197,
     165,   150,    71,    85,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,     2,     3,     0,
       0,     0,     0,   211,     0,     0,    84,   103,    88,   212,
     102,   188,   189,   190,   191,   192,   193,   216,     0,     0,
       0,   223,   222,   232,   261,   231,   218,   219,     0,     0,
       0,     0,   175,   163,   157,   155,   145,   146,     0,     0,
       0,     0,   104,     0,     0,    87,   109,   111,     0,     0,
     116,   110,   225,     0,   224,     0,     0,    46,    50,    45,
      49,    44,    48,    43,    47,    51,    52,     0,   260,     0,
     243,     0,    72,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,     0,     0,     0,     0,     0,     0,    26,
      27,    28,    29,     0,     0,     0,     0,    42,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
       0,     0,     0,     0,     0,    72,   235,     0,   257,   170,
     167,   166,   168,   169,   171,   174,     0,    80,    80,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,    99,
       0,     0,     0,     0,    80,    80,     0,     0,     0,   108,
     186,   115,   113,     0,   200,   201,   202,   205,   206,   207,
     203,   204,   198,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   209,   214,   213,   215,     0,   226,
       0,   247,   242,     0,    75,    75,   266,     0,     0,   255,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   172,     0,   160,   161,   130,   131,
       4,     5,   128,   129,   132,   123,   124,   127,     0,     0,
       0,     0,   126,   125,   158,   159,    86,    86,   112,   185,
     179,   182,   183,     0,     0,   105,   199,     0,     0,     0,
       0,     0,     0,   144,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   262,     0,   264,   259,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   173,     0,     0,    82,    80,   118,     0,
       0,   122,     0,   119,   106,   107,   178,   180,     0,    78,
     114,     0,   259,     0,     0,     0,     0,     0,   208,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   269,     0,     0,     0,   251,   252,     0,     0,     0,
       0,     0,   248,   249,     0,   267,     0,    77,    83,    81,
     117,   120,   121,   184,   181,    79,    73,     0,     0,     0,
       0,     0,     0,   143,     0,     0,     0,     0,     0,     0,
       0,   241,    75,    76,    75,   238,   258,     0,     0,     0,
       0,     0,   244,   245,   246,   241,     0,     0,   187,   210,
     134,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   240,     0,     0,   263,   265,     0,     0,     0,
     250,   253,   254,     0,   268,    74,     0,   140,     0,     0,
     136,   137,   138,   139,   133,     0,   229,     0,     0,     0,
     239,   236,     0,   256,     0,     0,     0,   227,     0,   228,
       0,     0,   237,   135,   141,   142,     0,     0,     0,     0,
       0,     0,   234,     0,     0,   233,     0,   230
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,    83,   302,   259,   260,   261,   262,   263,   230,   231,
     147,   232,    24,    15,    38,   458,   344,   366,   426,   296,
     367,    84,    85,   233,    87,    88,   128,   243,   333,   264,
     334,   118,     1,     2,    57,     3,    61,   212,    48,   113,
     216,    89,   377,   321,   322,   323,    39,    93,    16,    96,
      17,    29,    18,   327,   265,    90,   267,   445,    41,    42,
      43,   105,   106,   498,   107,   279,   472,   473,   205,   206,
     401,   207,   208
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -477
static const short int yypact[] =
{
    -477,    20,    54,   872,  -477,  -477,    86,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,    24,   110,    65,  -477,  -477,  -477,
    -477,    37,  -105,    51,     3,    -8,  -477,  -477,  -477,   110,
     111,   126,  -477,  -477,  -477,  -477,  -477,  -477,   703,   -23,
    -477,   -18,  -477,    -1,    23,    43,    64,    74,  -477,    70,
     111,   703,    71,    71,  -477,  -477,    71,    71,  -477,  -477,
    -477,  -477,  -477,    73,  -477,  -477,  -477,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,   188,
     189,   190,   457,  -477,   104,    77,  -477,  -477,   -29,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,   736,    80,
     118,  -477,  -477,  -477,  1196,  -477,  -477,  -477,   181,    87,
     192,   182,   183,  -477,  -477,  -477,  -477,  -477,   764,   764,
     802,   764,  -477,    81,    82,  -477,  -477,   -29,  -114,    83,
     513,  -477,    73,   988,  -477,   988,   988,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,  -477,   988,  -477,   764,
    -477,   158,   110,  -477,  -477,  -477,  -477,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,
    -477,  -477,  -477,   764,   764,   764,   764,   764,   764,  -477,
    -477,  -477,  -477,   764,   764,   764,   764,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,
     764,   764,   764,   764,   764,   110,  -477,    57,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,   -26,    89,    89,   160,
     170,   203,   172,   204,   174,   205,   178,   207,   206,   208,
     180,   210,   209,   862,    89,    89,   764,   764,   764,  -477,
     552,  -477,    96,    98,  -477,  -477,  -477,  -477,  -477,  -477,
    -477,  -477,   173,   102,   106,   108,   112,   114,   802,   115,
     116,   117,   119,   127,  -477,  -477,  -477,  -477,   113,  -477,
     125,  -477,  -477,   703,   130,   131,  -477,   988,   128,   133,
     134,   135,   138,   140,   142,   988,   988,   988,   143,   197,
     703,   764,   764,   212,  -477,    -2,  -477,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,  -477,  -477,   598,   802,
     485,   244,  -477,  -477,  -477,  -477,    15,   -82,   -29,  -477,
     104,  -477,   147,   145,   626,  -477,  -477,   250,   802,   802,
     802,   802,   802,  -477,  -113,   802,   802,   802,   802,   802,
     273,   274,   988,     0,  -477,    18,  -477,   163,   988,   156,
     764,   764,   764,   764,   764,   165,   167,   168,   764,   764,
     988,   988,   169,  -477,   271,   284,  -477,    89,  -477,   -22,
     -51,  -477,   -48,  -477,  -477,  -477,  -477,  -477,   665,   266,
    -477,   179,   163,   187,   211,   213,   215,   802,  -477,   216,
     217,   218,   219,   272,   988,   988,   186,   988,   292,   988,
     764,  -477,   220,   988,   222,  -477,  -477,   224,   225,   988,
     988,   988,  -477,  -477,   226,  -477,   764,  -477,  -477,  -477,
    -477,  -477,  -477,  -477,  -477,  -477,   265,   305,   223,   802,
     802,   802,   802,  -477,   802,   802,   802,   802,   764,   227,
     202,   764,   229,  -477,   229,  -477,   230,   988,   231,   764,
     764,   764,  -477,  -477,  -477,   764,   988,   319,  -477,  -477,
    -477,   233,   232,   234,   236,   235,   237,   238,   240,   242,
     323,    31,   230,   246,   282,  -477,  -477,   764,   241,   988,
    -477,  -477,  -477,   247,  -477,  -477,   802,  -477,   802,   802,
    -477,  -477,  -477,  -477,  -477,   988,  -477,  1105,    58,   303,
    -477,  -477,   245,  -477,   251,   252,   253,  -477,   239,  -477,
    1105,   368,  -477,  -477,  -477,  -477,   372,   258,   988,   988,
     374,   123,  -477,   988,   379,  -477,   988,  -477
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -477,  -477,  -477,   293,   298,   300,   301,   302,   -99,   -97,
    -440,  -477,   353,   383,  -134,  -477,  -271,    29,  -477,  -212,
    -477,   -49,  -477,   -38,  -477,   -68,   279,  -477,    -5,   177,
    -190,    97,  -477,  -477,  -477,  -477,   364,  -477,  -477,  -477,
    -477,     9,  -477,    40,  -477,  -477,   390,  -477,  -477,  -477,
    -477,  -477,   418,  -477,  -476,  -128,  -103,   101,  -477,   381,
    -477,   -93,  -477,  -477,  -477,  -477,    33,   -21,  -477,  -477,
      53,  -477,  -477
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -148
static const short int yytable[] =
{
      86,   145,   115,   146,   346,   266,   297,   266,   266,    91,
     397,    40,    25,    86,   127,    94,   238,   387,   273,   266,
       4,   508,   314,   315,    49,   388,   239,    19,   399,    20,
      51,   497,   268,   269,   517,   364,   137,   138,   139,   140,
     141,   142,   143,   144,   270,     7,     8,     9,    52,    11,
      53,    13,    25,    54,  -147,   365,   375,   398,   510,   131,
     133,    55,   127,   137,   138,   139,   140,   141,   142,   143,
     144,   290,    98,    99,   100,   398,   101,   102,   103,   387,
     217,   218,   387,   235,   135,     5,    30,   421,     6,    44,
      45,    46,   422,   129,    40,   136,     7,     8,     9,    10,
      11,    12,    13,   -86,   293,   313,   116,   117,   387,    47,
     294,   133,   131,    50,   420,   234,    92,    14,   369,   370,
     372,    58,    95,   137,   138,   139,   140,   141,   142,   143,
     144,    27,    19,    28,    20,   274,   275,   133,   277,   278,
     133,    60,   210,   211,    62,   133,   133,   133,   133,   266,
     119,   374,   108,   120,   121,   419,   131,   266,   266,   266,
     291,   292,   285,   286,   287,   133,   133,   496,   316,   317,
     318,   475,   109,   476,   347,    31,    32,    33,    34,    35,
      36,    37,   355,   356,   357,   298,   299,   -46,   -46,   -45,
     -45,   -44,   -44,   110,   509,   -43,   -43,   300,   301,   134,
     101,   102,   320,   111,   112,   -87,   122,   123,   124,   130,
     213,   209,   214,   215,   266,   240,   236,   237,   272,   295,
     266,   -50,   -49,   -48,   342,   -47,   324,   -53,   303,   -54,
     304,   325,   266,   266,   328,    86,   359,   326,   329,   396,
     330,   360,   363,   340,   331,   402,   332,   335,   336,   337,
     271,   338,    86,   361,   133,   341,   318,   414,   415,   339,
     343,   345,   348,   349,   350,   351,   266,   266,   352,   266,
     353,   266,   354,   358,   373,   266,   276,   378,   379,   280,
     381,   266,   266,   266,   281,   282,   283,   284,   394,   395,
     403,   439,   440,   400,   442,   409,   444,   410,   411,   416,
     448,   417,   418,   364,   288,   289,   452,   453,   454,   427,
     443,   438,   133,   405,   133,   133,   133,   429,   441,   266,
     133,   413,   457,   382,   383,   384,   385,   386,   266,   376,
     389,   390,   391,   392,   393,   459,   471,   485,   495,   398,
     320,   430,   511,   431,   478,   432,   434,   435,   436,   437,
     447,   266,   449,   484,   450,   451,   460,   470,   455,   474,
     477,   479,   133,   486,   488,   487,   489,   266,   490,   516,
     491,   492,   145,   493,   146,   494,   502,   501,   456,   499,
     503,   512,   433,   518,   513,   514,   515,   519,   520,   523,
     266,   266,   507,   362,   526,   266,   104,   200,   266,   145,
     469,   146,   201,   133,   202,   203,   204,    56,   425,   242,
     312,   133,   133,   133,   114,   521,   522,   133,   424,    59,
     525,    26,    97,   527,   461,   462,   463,   464,   524,   465,
     466,   467,   468,   446,   483,   428,     0,     0,     0,   133,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   404,     0,   406,   407,   408,     0,     0,     0,   412,
     125,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,     0,     0,    77,    78,     0,     0,
       0,   504,     0,   505,   506,    19,     0,    20,   125,   219,
     220,   221,   222,   223,   224,   225,   226,   227,   228,   229,
      75,    76,     0,     0,    77,    78,     0,     0,     0,     0,
       0,     0,     0,    19,     0,    20,   125,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
       0,     0,    77,    78,     0,     0,     0,     0,     0,     0,
       0,    19,     0,    20,     0,     0,     0,     0,     0,     0,
     480,   481,   482,   241,     0,   125,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,     0,
       0,    77,    78,     0,     0,     0,     0,     0,   500,     0,
      19,     0,    20,     0,     0,     0,     0,     0,    79,     0,
       0,    80,   319,     0,    81,     0,    82,   126,     0,     0,
       0,   125,   219,   220,   221,   222,   223,   224,   225,   226,
     227,   228,   229,    75,    76,     0,    79,    77,    78,    80,
       0,     0,    81,     0,    82,   371,    19,     0,    20,   125,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,     0,    79,    77,    78,    80,     0,     0,
      81,     0,    82,     0,    19,     0,    20,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   380,     0,   125,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,     0,    79,    77,    78,    80,     0,     0,    81,
       0,    82,     0,    19,     0,    20,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   423,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
       0,     0,    77,    78,     0,     0,     0,     0,     0,    79,
       0,    19,    80,    20,   368,    81,     0,    82,     0,   132,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,     0,     0,    77,    78,    79,     0,     0,
      80,     0,     0,    81,    19,    82,    20,   125,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,     0,     0,    77,    78,     0,     0,     0,     0,     0,
       0,     0,    19,     0,    20,     0,    79,     0,     0,    80,
       0,     0,    81,     0,    82,   125,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,    75,    76,     0,
       0,    77,    78,     0,     0,     0,     0,     0,     0,     0,
      19,     0,    20,     0,    79,     0,     0,    80,     0,     0,
      81,     0,    82,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    79,     0,     0,
      80,     0,     0,    81,     0,    82,     0,     0,     0,     0,
       0,    77,    78,     0,   305,   306,   307,     0,     0,     0,
      19,     0,    20,     0,     0,    79,     0,     0,    80,   -56,
      19,    81,    20,    82,     0,     0,     6,   -56,   -56,     0,
       0,     0,     0,     0,   -56,   -56,   -56,   -56,   -56,   -56,
     -56,     0,   -56,    21,     0,     0,     0,     0,     0,     0,
     -56,    22,     0,    79,     0,    23,    80,     0,     0,    81,
       0,    82,   153,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,     0,     0,     0,     0,     0,   253,     0,   254,
     179,   180,   181,   182,     0,   255,   256,   257,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,     0,     0,     0,     0,     0,   308,     0,     0,   309,
       0,   310,     0,     0,   311,   244,   245,    77,    78,   246,
     247,   248,   249,   250,   251,     0,    19,     0,    20,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     252,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165,   166,   167,   168,   169,   170,   171,   172,     0,     0,
       0,     0,     0,   253,     0,   254,   179,   180,   181,   182,
       0,   255,   256,   257,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   197,   198,   199,     0,     0,     0,
       0,     0,   244,   245,     0,   258,   246,   247,   248,   249,
     250,   251,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   252,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   153,   154,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,     0,     0,     0,     0,     0,
     253,     0,   254,   179,   180,   181,   182,     0,   255,   256,
     257,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   148,     0,     0,     0,     0,     0,
       0,     0,   258,     0,     0,   149,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   150,   151,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   152,     0,     0,     0,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   174,   175,     0,
       0,   176,   177,   178,   179,   180,   181,   182,   183,   184,
     185,   186,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   197,   198,   199
};

static const short int yycheck[] =
{
      38,   100,    51,   100,   275,   133,   218,   135,   136,    32,
      10,    29,     3,    51,    82,    33,   130,   130,   152,   147,
       0,   497,   234,   235,   129,   138,   140,    28,    10,    30,
      27,   471,   135,   136,   510,    37,     5,     6,     7,     8,
       9,    10,    11,    12,   147,    42,    43,    44,    45,    46,
      47,    48,    43,    50,     0,    57,   138,    57,   498,   141,
      98,    58,   130,     5,     6,     7,     8,     9,    10,    11,
      12,   205,    73,    74,    75,    57,    77,    78,    79,   130,
     118,   119,   130,   121,     4,    31,    62,   138,    34,    52,
      53,    54,   140,    84,    29,    15,    42,    43,    44,    45,
      46,    47,    48,   132,   130,   233,    35,    36,   130,    72,
     136,   149,   141,    62,   136,   120,   139,    63,   308,   309,
     310,   129,   140,     5,     6,     7,     8,     9,    10,    11,
      12,    45,    28,    47,    30,   173,   174,   175,   176,   177,
     178,    30,    55,    56,    18,   183,   184,   185,   186,   277,
      53,   136,   129,    56,    57,   367,   141,   285,   286,   287,
     103,   104,   200,   201,   202,   203,   204,   136,   236,   237,
     238,   442,   129,   444,   277,    65,    66,    67,    68,    69,
      70,    71,   285,   286,   287,    25,    26,    17,    18,    17,
      18,    17,    18,   129,   136,    17,    18,    17,    18,    98,
      77,    78,   240,   129,   134,   132,    18,    18,    18,   132,
      18,    30,    30,    30,   342,   132,   135,   135,    60,   130,
     348,    18,    18,    18,   273,    18,   130,    21,    18,    21,
      21,   133,   360,   361,   132,   273,    39,    64,   132,   342,
     132,   290,    30,   130,   132,   348,   132,   132,   132,   132,
     149,   132,   290,   291,   292,   130,   324,   360,   361,   132,
     130,   130,   134,   130,   130,   130,   394,   395,   130,   397,
     130,   399,   130,   130,    30,   403,   175,   130,   133,   178,
      30,   409,   410,   411,   183,   184,   185,   186,    15,    15,
     134,   394,   395,   130,   397,   130,   399,   130,   130,   130,
     403,    30,    18,    37,   203,   204,   409,   410,   411,   130,
      18,    39,   350,   351,   352,   353,   354,   130,   132,   447,
     358,   359,    57,   328,   329,   330,   331,   332,   456,   320,
     335,   336,   337,   338,   339,    30,   134,    18,    15,    57,
     378,   130,    39,   130,   447,   130,   130,   130,   130,   130,
     130,   479,   130,   456,   130,   130,   133,   130,   132,   130,
     130,   130,   400,   130,   130,   133,   130,   495,   133,   130,
     133,   133,   471,   133,   471,   133,   479,   136,   416,   133,
     133,   136,   387,    15,   133,   133,   133,    15,   130,    15,
     518,   519,   495,   292,    15,   523,    43,   104,   526,   498,
     438,   498,   104,   441,   104,   104,   104,    24,   379,   130,
     233,   449,   450,   451,    50,   518,   519,   455,   378,    29,
     523,     3,    41,   526,   429,   430,   431,   432,   521,   434,
     435,   436,   437,   400,   455,   382,    -1,    -1,    -1,   477,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   350,    -1,   352,   353,   354,    -1,    -1,    -1,   358,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    -1,    19,    20,    -1,    -1,
      -1,   486,    -1,   488,   489,    28,    -1,    30,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    28,    -1,    30,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,
     449,   450,   451,    40,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
      -1,    19,    20,    -1,    -1,    -1,    -1,    -1,   477,    -1,
      28,    -1,    30,    -1,    -1,    -1,    -1,    -1,   131,    -1,
      -1,   134,    40,    -1,   137,    -1,   139,   140,    -1,    -1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,   131,    19,    20,   134,
      -1,    -1,   137,    -1,   139,   140,    28,    -1,    30,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,   131,    19,    20,   134,    -1,    -1,
     137,    -1,   139,    -1,    28,    -1,    30,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    40,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    -1,   131,    19,    20,   134,    -1,    -1,   137,
      -1,   139,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    40,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,   131,
      -1,    28,   134,    30,   136,   137,    -1,   139,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,    -1,    19,    20,   131,    -1,    -1,
     134,    -1,    -1,   137,    28,   139,    30,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    28,    -1,    30,    -1,   131,    -1,    -1,   134,
      -1,    -1,   137,    -1,   139,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
      -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      28,    -1,    30,    -1,   131,    -1,    -1,   134,    -1,    -1,
     137,    -1,   139,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   131,    -1,    -1,
     134,    -1,    -1,   137,    -1,   139,    -1,    -1,    -1,    -1,
      -1,    19,    20,    -1,    22,    23,    24,    -1,    -1,    -1,
      28,    -1,    30,    -1,    -1,   131,    -1,    -1,   134,    27,
      28,   137,    30,   139,    -1,    -1,    34,    35,    36,    -1,
      -1,    -1,    -1,    -1,    42,    43,    44,    45,    46,    47,
      48,    -1,    50,    51,    -1,    -1,    -1,    -1,    -1,    -1,
      58,    59,    -1,   131,    -1,    63,   134,    -1,    -1,   137,
      -1,   139,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
     108,   109,   110,   111,    -1,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,    -1,    -1,    -1,    -1,    -1,   134,    -1,    -1,   137,
      -1,   139,    -1,    -1,   142,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    -1,    28,    -1,    30,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,   108,   109,   110,   111,
      -1,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,    -1,    -1,    -1,
      -1,    -1,    17,    18,    -1,   137,    21,    22,    23,    24,
      25,    26,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    62,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,   108,   109,   110,   111,    -1,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,    38,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   137,    -1,    -1,    49,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    60,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    76,    -1,    -1,    -1,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,    -1,
      -1,   105,   106,   107,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   175,   176,   178,     0,    31,    34,    42,    43,    44,
      45,    46,    47,    48,    63,   156,   191,   193,   195,    28,
      30,    51,    59,    63,   155,   184,   195,    45,    47,   194,
      62,    65,    66,    67,    68,    69,    70,    71,   157,   189,
      29,   201,   202,   203,    52,    53,    54,    72,   181,   129,
      62,    27,    45,    47,    50,    58,   156,   177,   129,   189,
      30,   179,    18,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    19,    20,   131,
     134,   137,   139,   144,   164,   165,   166,   167,   168,   184,
     198,    32,   139,   190,    33,   140,   192,   202,    73,    74,
      75,    77,    78,    79,   155,   204,   205,   207,   129,   129,
     129,   129,   134,   182,   179,   164,    35,    36,   174,   174,
     174,   174,    18,    18,    18,     3,   140,   168,   169,   184,
     132,   141,     3,   166,   200,     4,    15,     5,     6,     7,
       8,     9,    10,    11,    12,   151,   152,   153,    38,    49,
      60,    61,    76,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     146,   147,   148,   149,   150,   211,   212,   214,   215,    30,
      55,    56,   180,    18,    30,    30,   183,   166,   166,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
     151,   152,   154,   166,   171,   166,   135,   135,   130,   140,
     132,    40,   169,   170,    17,    18,    21,    22,    23,    24,
      25,    26,    62,   105,   107,   113,   114,   115,   137,   146,
     147,   148,   149,   150,   172,   197,   198,   199,   199,   199,
     199,   200,    60,   157,   166,   166,   200,   166,   166,   208,
     200,   200,   200,   200,   200,   166,   166,   166,   200,   200,
     157,   103,   104,   130,   136,   130,   162,   162,    25,    26,
      17,    18,   145,    18,    21,    22,    23,    24,   134,   137,
     139,   142,   172,   198,   162,   162,   168,   168,   168,    40,
     166,   186,   187,   188,   130,   133,    64,   196,   132,   132,
     132,   132,   132,   171,   173,   132,   132,   132,   132,   132,
     130,   130,   164,   130,   159,   130,   159,   199,   134,   130,
     130,   130,   130,   130,   130,   199,   199,   199,   130,    39,
     164,   166,   200,    30,    37,    57,   160,   163,   136,   173,
     173,   140,   173,    30,   136,   138,   184,   185,   130,   133,
      40,    30,   171,   171,   171,   171,   171,   130,   138,   171,
     171,   171,   171,   171,    15,    15,   199,    10,    57,    10,
     130,   213,   199,   134,   200,   166,   200,   200,   200,   130,
     130,   130,   200,   166,   199,   199,   130,    30,    18,   162,
     136,   138,   140,    40,   186,   160,   161,   130,   213,   130,
     130,   130,   130,   171,   130,   130,   130,   130,    39,   199,
     199,   132,   199,    18,   199,   200,   209,   130,   199,   130,
     130,   130,   199,   199,   199,   132,   166,    57,   158,    30,
     133,   171,   171,   171,   171,   171,   171,   171,   171,   166,
     130,   134,   209,   210,   130,   159,   159,   130,   199,   130,
     200,   200,   200,   210,   199,    18,   130,   133,   130,   130,
     133,   133,   133,   133,   133,    15,   136,   153,   206,   133,
     200,   136,   199,   133,   171,   171,   171,   199,   197,   136,
     153,    39,   136,   133,   133,   133,   130,   197,    15,    15,
     130,   199,   199,    15,   204,   199,    15,   199
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
        case 55:
#line 321 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 56:
#line 324 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string(""); 
  ;}
    break;

  case 64:
#line 331 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 71:
#line 336 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += *(yyvsp[0].String); 
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
    ;}
    break;

  case 72:
#line 341 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 73:
#line 346 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 74:
#line 347 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { *(yyvsp[-1].String) += " " + *(yyvsp[0].String); delete (yyvsp[0].String); (yyval.String) = (yyvsp[-1].String); ;}
    break;

  case 75:
#line 350 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 76:
#line 351 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, ", "); 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 77:
#line 359 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 78:
#line 365 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 80:
#line 369 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 81:
#line 370 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      (yyvsp[-1].String)->insert(0, ", ");
      if (!(yyvsp[0].String)->empty())
        *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 83:
#line 380 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
      *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 101:
#line 402 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type).newTy = (yyvsp[0].String); 
    (yyval.Type).oldTy = OpaqueTy; 
  ;}
    break;

  case 102:
#line 406 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type).newTy = (yyvsp[0].String);
    (yyval.Type).oldTy = UnresolvedTy;
  ;}
    break;

  case 103:
#line 410 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type) = (yyvsp[0].Type); 
  ;}
    break;

  case 104:
#line 413 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Type UpReference
    (yyvsp[0].String)->insert(0, "\\");
    (yyval.Type).newTy = (yyvsp[0].String);
    (yyval.Type).oldTy = NumericTy;
  ;}
    break;

  case 105:
#line 418 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {           // Function derived type?
    *(yyvsp[-3].Type).newTy += "( " + *(yyvsp[-1].String) + " )";
    delete (yyvsp[-1].String);
    (yyval.Type).newTy = (yyvsp[-3].Type).newTy;
    (yyval.Type).oldTy = FunctionTy;
  ;}
    break;

  case 106:
#line 424 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Sized array type?
    (yyvsp[-3].String)->insert(0,"[ ");
    *(yyvsp[-3].String) += " x " + *(yyvsp[-1].Type).newTy + " ]";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].String);
    (yyval.Type).oldTy = ArrayTy;
    (yyval.Type).elemTy = (yyvsp[-1].Type).oldTy;
  ;}
    break;

  case 107:
#line 432 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {          // Packed array type?
    (yyvsp[-3].String)->insert(0,"< ");
    *(yyvsp[-3].String) += " x " + *(yyvsp[-1].Type).newTy + " >";
    delete (yyvsp[-1].Type).newTy;
    (yyval.Type).newTy = (yyvsp[-3].String);
    (yyval.Type).oldTy = PackedTy;
    (yyval.Type).elemTy = (yyvsp[-1].Type).oldTy;
  ;}
    break;

  case 108:
#line 440 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                        // Structure type?
    (yyvsp[-1].String)->insert(0, "{ ");
    *(yyvsp[-1].String) += " }";
    (yyval.Type).newTy = (yyvsp[-1].String);
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 109:
#line 446 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                  // Empty structure type?
    (yyval.Type).newTy = new std::string("{}");
    (yyval.Type).oldTy = StructTy;
  ;}
    break;

  case 110:
#line 450 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                             // Pointer type?
    *(yyvsp[-1].Type).newTy += '*';
    (yyval.Type).elemTy = (yyvsp[-1].Type).oldTy;
    (yyvsp[-1].Type).oldTy = PointerTy;
    (yyval.Type) = (yyvsp[-1].Type);
  ;}
    break;

  case 111:
#line 461 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].Type).newTy;
  ;}
    break;

  case 112:
#line 464 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Type).newTy;
    delete (yyvsp[0].Type).newTy;
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 114:
#line 473 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 115:
#line 478 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 116:
#line 481 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 117:
#line 491 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " [ " + *(yyvsp[-1].String) + " ]";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 118:
#line 497 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += "[ ]";
  ;}
    break;

  case 119:
#line 502 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " c" + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 120:
#line 508 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " < " + *(yyvsp[-1].String) + " >";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 121:
#line 514 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-3].Type).newTy);
    *(yyval.Const).cnst += " { " + *(yyvsp[-1].String) + " }";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 122:
#line 520 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-2].Type).newTy);
    *(yyval.Const).cnst += " {}";
  ;}
    break;

  case 123:
#line 525 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst +=  " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 124:
#line 531 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 125:
#line 537 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 126:
#line 543 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 127:
#line 549 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 128:
#line 555 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {      // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 129:
#line 561 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {            // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 130:
#line 567 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                      // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 131:
#line 573 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                     // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 132:
#line 579 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Float & Double constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string(*(yyvsp[-1].Type).newTy);
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 133:
#line 587 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 134:
#line 601 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 135:
#line 613 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 136:
#line 618 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* op = getDivRemOpcode(*(yyvsp[-5].String), (yyvsp[-3].Const).type); 
    (yyval.String) = new std::string(op);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
  ;}
    break;

  case 137:
#line 624 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 138:
#line 629 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 139:
#line 634 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-5].String)->c_str();
    if (*(yyvsp[-5].String) == "shr")
      shiftop = ((yyvsp[-3].Const).type.isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
  ;}
    break;

  case 140:
#line 642 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 141:
#line 647 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 142:
#line 652 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 143:
#line 662 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 144:
#line 667 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Const).cnst); (yyvsp[0].Const).destroy(); ;}
    break;

  case 147:
#line 682 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
;}
    break;

  case 148:
#line 687 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 149:
#line 690 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 150:
#line 695 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "module asm " << " " << *(yyvsp[0].String) << "\n";
    (yyval.String) = 0;
  ;}
    break;

  case 151:
#line 699 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "implementation\n";
    (yyval.String) = 0;
  ;}
    break;

  case 152:
#line 703 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = 0; ;}
    break;

  case 154:
#line 705 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].String); *(yyval.String) = "external"; ;}
    break;

  case 155:
#line 708 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    EnumeratedTypes.push_back((yyvsp[0].Type));
    if (!(yyvsp[-2].String)->empty()) {
      NamedTypes[*(yyvsp[-2].String)].newTy = new std::string(*(yyvsp[0].Type).newTy);
      NamedTypes[*(yyvsp[-2].String)].oldTy = (yyvsp[0].Type).oldTy;
      NamedTypes[*(yyvsp[-2].String)].elemTy = (yyvsp[0].Type).elemTy;
      *O << *(yyvsp[-2].String) << " = ";
    }
    *O << "type " << *(yyvsp[0].Type).newTy << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 156:
#line 720 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {       // Function prototypes can be in const pool
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 157:
#line 725 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  // Asm blocks can be in the const pool
    *O << *(yyvsp[-2].String) << " " << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 158:
#line 730 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Const).cnst << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Const).destroy(); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 159:
#line 737 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O <<  *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 160:
#line 744 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 161:
#line 751 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty())
      *O << *(yyvsp[-4].String) << " = ";
    *O << *(yyvsp[-3].String) << " " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 162:
#line 758 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 163:
#line 763 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " = " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 164:
#line 768 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = 0;
  ;}
    break;

  case 168:
#line 778 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 169:
#line 783 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    if (*(yyvsp[0].String) == "64")
      SizeOfPointer = 64;
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 170:
#line 790 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 171:
#line 795 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 172:
#line 802 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1].String)->insert(0, "[ ");
    *(yyvsp[-1].String) += " ]";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 173:
#line 809 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 175:
#line 815 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 179:
#line 824 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 180:
#line 826 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  (yyval.String) = (yyvsp[-1].Type).newTy;
  if (!(yyvsp[0].String)->empty())
    *(yyval.String) += " " + *(yyvsp[0].String);
  delete (yyvsp[0].String);
;}
    break;

  case 181:
#line 833 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 182:
#line 837 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 183:
#line 841 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 184:
#line 844 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    (yyval.String) = (yyvsp[-2].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 185:
#line 849 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 186:
#line 852 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 187:
#line 855 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 188:
#line 874 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("{"); delete (yyvsp[0].String); ;}
    break;

  case 189:
#line 875 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string ("{"); ;}
    break;

  case 190:
#line 877 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if (!(yyvsp[-2].String)->empty()) {
    *O << *(yyvsp[-2].String) << " ";
  }
  *O << *(yyvsp[-1].String) << " " << *(yyvsp[0].String) << "\n";
  delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String);
  (yyval.String) = 0;
;}
    break;

  case 191:
#line 886 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); delete (yyvsp[0].String); ;}
    break;

  case 192:
#line 887 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); ;}
    break;

  case 193:
#line 889 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
  if ((yyvsp[-1].String))
    *O << *(yyvsp[-1].String);
  *O << '\n' << *(yyvsp[0].String) << "\n";
  (yyval.String) = 0;
;}
    break;

  case 194:
#line 897 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 197:
#line 903 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[-1].String);
    *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 198:
#line 916 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 208:
#line 922 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, "<");
    *(yyvsp[-1].String) += ">";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 210:
#line 928 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty()) {
      *(yyvsp[-4].String) += " " + *(yyvsp[-3].String);
    }
    *(yyvsp[-4].String) += " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 213:
#line 941 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).val = (yyvsp[0].String);
    (yyval.Value).constant = false;
    (yyval.Value).type.newTy = 0;
    (yyval.Value).type.oldTy = UnresolvedTy;
  ;}
    break;

  case 214:
#line 947 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).val = (yyvsp[0].String);
    (yyval.Value).constant = true;
    (yyval.Value).type.newTy = 0;
    (yyval.Value).type.oldTy = UnresolvedTy;
  ;}
    break;

  case 215:
#line 958 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value) = (yyvsp[0].Value);
    (yyval.Value).type = (yyvsp[-1].Type);
    (yyval.Value).val->insert(0, *(yyvsp[-1].Type).newTy + " ");
  ;}
    break;

  case 216:
#line 964 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 217:
#line 967 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { // Do not allow functions with 0 basic blocks   
    (yyval.String) = 0;
  ;}
    break;

  case 218:
#line 975 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 219:
#line 979 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 220:
#line 984 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 221:
#line 987 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 223:
#line 993 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].String); *(yyval.String) = "unwind"; ;}
    break;

  case 224:
#line 995 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {              // Return with a result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 225:
#line 1000 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                                       // Return with no result...
    *O << "    " << *(yyvsp[-1].String) << " " << *(yyvsp[0].Type).newTy << "\n";
    delete (yyvsp[-1].String); (yyvsp[0].Type).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 226:
#line 1005 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {                         // Unconditional Branch...
    *O << "    " << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 227:
#line 1010 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].Value).val << ", " 
       << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].Value).val << ", " << *(yyvsp[-1].Type).newTy << " " 
       << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); (yyvsp[-6].Value).destroy(); (yyvsp[-4].Type).destroy(); (yyvsp[-3].Value).destroy(); 
    (yyvsp[-1].Type).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 228:
#line 1018 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-8].String) << " " << *(yyvsp[-7].Type).newTy << " " << *(yyvsp[-6].Value).val << ", " 
       << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].Value).val << " [" << *(yyvsp[-1].String) << " ]\n";
    delete (yyvsp[-8].String); (yyvsp[-7].Type).destroy(); (yyvsp[-6].Value).destroy(); (yyvsp[-4].Type).destroy(); (yyvsp[-3].Value).destroy(); 
    delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 229:
#line 1025 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[-7].String) << " " << *(yyvsp[-6].Type).newTy << " " << *(yyvsp[-5].Value).val << ", " 
       << *(yyvsp[-3].Type).newTy << " " << *(yyvsp[-2].Value).val << "[]\n";
    delete (yyvsp[-7].String); (yyvsp[-6].Type).destroy(); (yyvsp[-5].Value).destroy(); (yyvsp[-3].Type).destroy(); (yyvsp[-2].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 230:
#line 1032 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    ";
    if (!(yyvsp[-13].String)->empty())
      *O << *(yyvsp[-13].String) << " = ";
    *O << *(yyvsp[-12].String) << " " << *(yyvsp[-11].String) << " " << *(yyvsp[-10].Type).newTy << " " << *(yyvsp[-9].Value).val << " (";
    for (unsigned i = 0; i < (yyvsp[-7].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-7].ValList))[i];
      *O << *VI.val;
      if (i+1 < (yyvsp[-7].ValList)->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *(yyvsp[-5].String) << " " << *(yyvsp[-4].Type).newTy << " " << *(yyvsp[-3].Value).val << " " 
       << *(yyvsp[-2].String) << " " << *(yyvsp[-1].Type).newTy << " " << *(yyvsp[0].Value).val << "\n";
    delete (yyvsp[-13].String); delete (yyvsp[-12].String); delete (yyvsp[-11].String); (yyvsp[-10].Type).destroy(); (yyvsp[-9].Value).destroy(); delete (yyvsp[-7].ValList); 
    delete (yyvsp[-5].String); (yyvsp[-4].Type).destroy(); (yyvsp[-3].Value).destroy(); delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); 
    (yyvsp[0].Value).destroy(); 
    (yyval.String) = 0;
  ;}
    break;

  case 231:
#line 1051 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 232:
#line 1056 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << "\n";
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 233:
#line 1062 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].String) + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Type).destroy(); delete (yyvsp[-3].String); (yyvsp[-1].Type).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 234:
#line 1067 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-3].String)->insert(0, *(yyvsp[-4].Type).newTy + " " );
    *(yyvsp[-3].String) += ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Type).destroy(); (yyvsp[-1].Type).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 235:
#line 1075 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-1].String) += " = ";
    *(yyvsp[-1].String) += *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
  ;}
    break;

  case 236:
#line 1084 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {    // Used for PHI nodes
    (yyvsp[-3].Value).val->insert(0, *(yyvsp[-5].Type).newTy + "[");
    *(yyvsp[-3].Value).val += "," + *(yyvsp[-1].Value).val + "]";
    (yyvsp[-5].Type).destroy(); (yyvsp[-1].Value).destroy();
    (yyval.String) = new std::string(*(yyvsp[-3].Value).val);
    (yyvsp[-3].Value).destroy();
  ;}
    break;

  case 237:
#line 1091 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6].String) += ", [" + *(yyvsp[-3].Value).val + "," + *(yyvsp[-1].Value).val + "]";
    (yyvsp[-3].Value).destroy(); (yyvsp[-1].Value).destroy();
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 238:
#line 1099 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.ValList) = new ValueList();
    (yyval.ValList)->push_back((yyvsp[0].Value));
  ;}
    break;

  case 239:
#line 1103 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-2].ValList)->push_back((yyvsp[0].Value));
    (yyval.ValList) = (yyvsp[-2].ValList);
  ;}
    break;

  case 240:
#line 1110 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = (yyvsp[0].ValList); ;}
    break;

  case 241:
#line 1111 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = new ValueList(); ;}
    break;

  case 242:
#line 1115 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 244:
#line 1123 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* op = getDivRemOpcode(*(yyvsp[-4].String), (yyvsp[-3].Type)); 
    (yyval.String) = new std::string(op);
    *(yyval.String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    delete (yyvsp[-4].String); (yyvsp[-3].Type).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 245:
#line 1129 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-3].Type).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 246:
#line 1134 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-4].String) += " " + *(yyvsp[-3].Type).newTy + " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-3].Type).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 247:
#line 1139 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 248:
#line 1144 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-3].String)->c_str();
    if (*(yyvsp[-3].String) == "shr")
      shiftop = ((yyvsp[-2].Value).type.isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 249:
#line 1152 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
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

  case 250:
#line 1166 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 251:
#line 1171 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Type).newTy;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Type).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 252:
#line 1176 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 253:
#line 1181 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 254:
#line 1186 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyvsp[-4].Value).destroy(); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 255:
#line 1191 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 256:
#line 1196 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-6].String) += " " + *(yyvsp[-5].String);
    if (!(yyvsp[-6].String)->empty())
      *(yyvsp[-6].String) += " ";
    *(yyvsp[-6].String) += *(yyvsp[-4].Type).newTy + " " + *(yyvsp[-3].Value).val + "(";
    for (unsigned i = 0; i < (yyvsp[-1].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-1].ValList))[i];
      *(yyvsp[-6].String) += *VI.val;
      if (i+1 < (yyvsp[-1].ValList)->size())
        *(yyvsp[-6].String) += ", ";
      VI.destroy();
    }
    *(yyvsp[-6].String) += ")";
    delete (yyvsp[-5].String); (yyvsp[-4].Type).destroy(); (yyvsp[-3].Value).destroy(); delete (yyvsp[-1].ValList);
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 258:
#line 1218 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = (yyvsp[0].ValList); ;}
    break;

  case 259:
#line 1219 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {  (yyval.ValList) = new ValueList(); ;}
    break;

  case 261:
#line 1224 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 262:
#line 1227 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 263:
#line 1234 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].Value).val;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); (yyvsp[-1].Value).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 264:
#line 1241 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + *(yyvsp[-1].Type).newTy;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyvsp[-1].Type).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 265:
#line 1248 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Type).newTy + ", " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].Value).val;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyvsp[-4].Type).destroy(); (yyvsp[-2].Type).destroy(); (yyvsp[-1].Value).destroy(); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 266:
#line 1255 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 267:
#line 1260 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty())
      *(yyvsp[-3].String) += " ";
    *(yyvsp[-3].String) += *(yyvsp[-2].String) + " " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].Value).val;
    delete (yyvsp[-2].String); (yyvsp[-1].Type).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 268:
#line 1267 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-5].String) += " ";
    *(yyvsp[-5].String) += *(yyvsp[-4].String) + " " + *(yyvsp[-3].Value).val + ", " + *(yyvsp[-1].Type).newTy + " " + *(yyvsp[0].Value).val;
    delete (yyvsp[-4].String); (yyvsp[-3].Value).destroy(); (yyvsp[-1].Type).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 269:
#line 1274 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"
    {
    // Upgrade the indices
    for (unsigned i = 0; i < (yyvsp[0].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[0].ValList))[i];
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
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Type).newTy + " " + *(yyvsp[-1].Value).val;
    for (unsigned i = 0; i < (yyvsp[0].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[0].ValList))[i];
      *(yyvsp[-3].String) += ", " + *VI.val;
      VI.destroy();
    }
    (yyvsp[-2].Type).destroy(); (yyvsp[-1].Value).destroy(); delete (yyvsp[0].ValList);
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 3665 "UpgradeParser.tab.c"

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


#line 1298 "/proj/llvm/llvm-4/tools/llvm-upgrade/UpgradeParser.y"


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

