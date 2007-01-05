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
     ICMP = 355,
     FCMP = 356,
     EQ = 357,
     NE = 358,
     SLT = 359,
     SGT = 360,
     SLE = 361,
     SGE = 362,
     OEQ = 363,
     ONE = 364,
     OLT = 365,
     OGT = 366,
     OLE = 367,
     OGE = 368,
     ORD = 369,
     UNO = 370,
     UEQ = 371,
     UNE = 372,
     ULT = 373,
     UGT = 374,
     ULE = 375,
     UGE = 376,
     MALLOC = 377,
     ALLOCA = 378,
     FREE = 379,
     LOAD = 380,
     STORE = 381,
     GETELEMENTPTR = 382,
     PHI_TOK = 383,
     SELECT = 384,
     SHL = 385,
     SHR = 386,
     ASHR = 387,
     LSHR = 388,
     VAARG = 389,
     EXTRACTELEMENT = 390,
     INSERTELEMENT = 391,
     SHUFFLEVECTOR = 392,
     CAST = 393,
     TRUNC = 394,
     ZEXT = 395,
     SEXT = 396,
     FPTRUNC = 397,
     FPEXT = 398,
     FPTOUI = 399,
     FPTOSI = 400,
     UITOFP = 401,
     SITOFP = 402,
     PTRTOINT = 403,
     INTTOPTR = 404,
     BITCAST = 405
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
#define ICMP 355
#define FCMP 356
#define EQ 357
#define NE 358
#define SLT 359
#define SGT 360
#define SLE 361
#define SGE 362
#define OEQ 363
#define ONE 364
#define OLT 365
#define OGT 366
#define OLE 367
#define OGE 368
#define ORD 369
#define UNO 370
#define UEQ 371
#define UNE 372
#define ULT 373
#define UGT 374
#define ULE 375
#define UGE 376
#define MALLOC 377
#define ALLOCA 378
#define FREE 379
#define LOAD 380
#define STORE 381
#define GETELEMENTPTR 382
#define PHI_TOK 383
#define SELECT 384
#define SHL 385
#define SHR 386
#define ASHR 387
#define LSHR 388
#define VAARG 389
#define EXTRACTELEMENT 390
#define INSERTELEMENT 391
#define SHUFFLEVECTOR 392
#define CAST 393
#define TRUNC 394
#define ZEXT 395
#define SEXT 396
#define FPTRUNC 397
#define FPEXT 398
#define FPTOUI 399
#define FPTOSI 400
#define UITOFP 401
#define SITOFP 402
#define PTRTOINT 403
#define INTTOPTR 404
#define BITCAST 405




/* Copy the first part of user declarations.  */
#line 14 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"

#include "UpgradeInternals.h"
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

typedef std::vector<const TypeInfo*> TypeVector;
static TypeVector EnumeratedTypes;
typedef std::map<std::string,const TypeInfo*> TypeMap;
static TypeMap NamedTypes;
typedef std::map<const TypeInfo*,std::string> TypePlaneMap;
typedef std::map<std::string,TypePlaneMap> GlobalsTypeMap;
static GlobalsTypeMap Globals;

static void warning(const std::string& msg);

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

TypeInfo::TypeRegMap TypeInfo::registry;

const TypeInfo* TypeInfo::get(const std::string &newType, Types oldType) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, Types oldType, 
                              const TypeInfo* eTy, const TypeInfo* rTy) {
  TypeInfo* Ty= new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  Ty->elemTy = const_cast<TypeInfo*>(eTy);
  Ty->resultTy = const_cast<TypeInfo*>(rTy);
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, Types oldType, 
                              const TypeInfo *eTy, uint64_t elems) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  Ty->elemTy = const_cast<TypeInfo*>(eTy);
  Ty->nelems = elems;
  return  add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, Types oldType, 
                              TypeList* TL) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = oldType;
  Ty->elements = TL;
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::get(const std::string& newType, const TypeInfo* resTy,
                              TypeList* TL) {
  TypeInfo* Ty = new TypeInfo();
  Ty->newTy = newType;
  Ty->oldTy = FunctionTy;
  Ty->resultTy = const_cast<TypeInfo*>(resTy);
  Ty->elements = TL;
  return add_new_type(Ty);
}

const TypeInfo* TypeInfo::resolve() const {
  if (isUnresolved()) {
    if (getNewTy()[0] == '%' && isdigit(getNewTy()[1])) {
      unsigned ref = atoi(&((getNewTy().c_str())[1])); // skip the %
      if (ref < EnumeratedTypes.size()) {
        return EnumeratedTypes[ref];
      } else {
        std::string msg("Can't resolve numbered type: ");
        msg += getNewTy();
        yyerror(msg.c_str());
      }
    } else {
      TypeMap::iterator I = NamedTypes.find(getNewTy());
      if (I != NamedTypes.end()) {
        return I->second;
      } else {
        std::string msg("Cannot resolve type: ");
        msg += getNewTy();
        yyerror(msg.c_str());
      }
    }
  }
  // otherwise its already resolved.
  return this;
}

bool TypeInfo::operator<(const TypeInfo& that) const {
  if (this == &that)
    return false;
  if (oldTy != that.oldTy)
    return oldTy < that.oldTy;
  switch (oldTy) {
    case UpRefTy: {
      unsigned thisUp = this->getUpRefNum();
      unsigned thatUp = that.getUpRefNum();
      return thisUp < thatUp;
    }
    case PackedTy:
    case ArrayTy:
      if (this->nelems != that.nelems)
        return nelems < that.nelems;
    case PointerTy: {
      const TypeInfo* thisTy = this->elemTy;
      const TypeInfo* thatTy = that.elemTy;
      return *thisTy < *thatTy;
    }
    case FunctionTy: {
      const TypeInfo* thisTy = this->resultTy;
      const TypeInfo* thatTy = that.resultTy;
      if (!thisTy->sameOldTyAs(thatTy))
        return *thisTy < *thatTy;
      /* FALL THROUGH */
    }
    case StructTy:
    case PackedStructTy: {
      if (elements->size() != that.elements->size())
        return elements->size() < that.elements->size();
      for (unsigned i = 0; i < elements->size(); i++) {
        const TypeInfo* thisTy = (*this->elements)[i];
        const TypeInfo* thatTy = (*that.elements)[i];
        if (!thisTy->sameOldTyAs(thatTy))
          return *thisTy < *thatTy;
      }
      break;
    }
    case UnresolvedTy:
      return this->newTy < that.newTy;
    default:
      break;
  }
  return false; 
}

bool TypeInfo::sameOldTyAs(const TypeInfo* that) const {
  if (that == 0)
    return false;
  if ( this == that ) 
    return true;
  if (oldTy != that->oldTy)
    return false;
  switch (oldTy) {
    case PackedTy:
    case ArrayTy:
      if (nelems != that->nelems)
        return false;
      /* FALL THROUGH */
    case PointerTy: {
      const TypeInfo* thisTy = this->elemTy;
      const TypeInfo* thatTy = that->elemTy;
      return thisTy->sameOldTyAs(thatTy);
    }
    case FunctionTy: {
      const TypeInfo* thisTy = this->resultTy;
      const TypeInfo* thatTy = that->resultTy;
      if (!thisTy->sameOldTyAs(thatTy))
        return false;
      /* FALL THROUGH */
    }
    case StructTy:
    case PackedStructTy: {
      if (elements->size() != that->elements->size())
        return false;
      for (unsigned i = 0; i < elements->size(); i++) {
        const TypeInfo* thisTy = (*this->elements)[i];
        const TypeInfo* thatTy = (*that->elements)[i];
        if (!thisTy->sameOldTyAs(thatTy))
          return false;
      }
      return true;
    }
    case UnresolvedTy:
      return this->newTy == that->newTy;
    default:
      return true; // for all others oldTy == that->oldTy is sufficient
  }
  return true;
}

bool TypeInfo::isUnresolvedDeep() const {
  switch (oldTy) {
    case UnresolvedTy: 
      return true;
    case PackedTy:
    case ArrayTy:
    case PointerTy:
      return elemTy->isUnresolvedDeep();
    case PackedStructTy:
    case StructTy:
      for (unsigned i = 0; i < elements->size(); i++)
        if ((*elements)[i]->isUnresolvedDeep())
          return true;
      return false;
    default:
      return false;
  }
}

unsigned TypeInfo::getBitWidth() const {
  switch (oldTy) {
    default:
    case LabelTy:
    case VoidTy : return 0;
    case BoolTy : return 1;
    case SByteTy: case UByteTy : return 8;
    case ShortTy: case UShortTy : return 16;
    case IntTy: case UIntTy: case FloatTy: return 32;
    case LongTy: case ULongTy: case DoubleTy : return 64;
    case PointerTy: return SizeOfPointer; // global var
    case PackedTy: 
    case ArrayTy: 
      return nelems * elemTy->getBitWidth();
    case StructTy:
    case PackedStructTy: {
      uint64_t size = 0;
      for (unsigned i = 0; i < elements->size(); i++) {
        size += (*elements)[i]->getBitWidth();
      }
      return size;
    }
  }
}

const TypeInfo* TypeInfo::getIndexedType(const ValueInfo&  VI) const {
  if (isStruct()) {
    if (VI.isConstant() && VI.type->isInteger()) {
      size_t pos = VI.val->find(' ') + 1;
      if (pos < VI.val->size()) {
        uint64_t idx = atoi(VI.val->substr(pos).c_str());
        return (*elements)[idx];
      } else {
        yyerror("Invalid value for constant integer");
        return 0;
      }
    } else {
      yyerror("Structure requires constant index");
      return 0;
    }
  }
  if (isArray() || isPacked() || isPointer())
    return elemTy;
  yyerror("Invalid type for getIndexedType");
  return 0;
}

TypeInfo& TypeInfo::operator=(const TypeInfo& that) {
  oldTy = that.oldTy;
  nelems = that.nelems;
  newTy = that.newTy;
  elemTy = that.elemTy;
  resultTy = that.resultTy;
  if (that.elements) {
    elements = new TypeList(that.elements->size());
    *elements = *that.elements;
  } else {
    elements = 0;
  }
  return *this;
}

const TypeInfo* TypeInfo::add_new_type(TypeInfo* newTy) {
  TypeRegMap::iterator I = registry.find(newTy);
  if (I != registry.end()) {
    delete newTy;
    return *I;
  }
  registry.insert(newTy);
  return newTy;
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

static std::string getCastUpgrade(const std::string& Src, const TypeInfo* SrcTy,
                                  const TypeInfo* DstTy, bool isConst)
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
    SrcTy = TypeInfo::get("i64", ULongTy);
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
  SrcTy = SrcTy->resolve();
  DstTy = DstTy->resolve();
  std::string Opcode(getCastOpcode(Source, SrcTy, DstTy));
  if (isConst)
    Result += Opcode + "( " + Source + " to " + DstTy->getNewTy() + ")";
  else
    Result += Opcode + " " + Source + " to " + DstTy->getNewTy();
  return Result;
}

const char* getDivRemOpcode(const std::string& opcode, const TypeInfo* TI) {
  const char* op = opcode.c_str();
  const TypeInfo* Ty = TI->resolve();
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

static const TypeInfo* getFunctionReturnType(const TypeInfo* PFTy) {
  PFTy = PFTy->resolve();
  if (PFTy->isPointer()) {
    const TypeInfo* ElemTy = PFTy->getElementType();
    ElemTy = ElemTy->resolve();
    if (ElemTy->isFunction())
      return ElemTy->getResultType();
  } else if (PFTy->isFunction()) {
    return PFTy->getResultType();
  }
  return PFTy;
}

typedef std::vector<const TypeInfo*> UpRefStack;
static const TypeInfo* ResolveUpReference(const TypeInfo* Ty, 
                                          UpRefStack* stack) {
  assert(Ty->isUpReference() && "Can't resolve a non-upreference");
  unsigned upref = Ty->getUpRefNum();
  assert(upref < stack->size() && "Invalid up reference");
  return (*stack)[upref - stack->size() - 1];
}

static const TypeInfo* getGEPIndexedType(const TypeInfo* PTy, ValueList* idxs) {
  const TypeInfo* Result = PTy = PTy->resolve();
  assert(PTy->isPointer() && "GEP Operand is not a pointer?");
  UpRefStack stack;
  for (unsigned i = 0; i < idxs->size(); ++i) {
    if (Result->isComposite()) {
      Result = Result->getIndexedType((*idxs)[i]);
      Result = Result->resolve();
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
static std::string getUniqueName(const std::string *Name, const TypeInfo* Ty,
                                 bool isGlobal = false) {

  // If its not a symbolic name, don't modify it, probably a constant val.
  if ((*Name)[0] != '%' && (*Name)[0] != '"')
    return *Name;

  // If its a numeric reference, just leave it alone.
  if (isdigit((*Name)[1]))
    return *Name;

  // Resolve the type
  Ty = Ty->resolve(); 

  // If its a global name, get its uniquified name, if any
  GlobalsTypeMap::iterator GI = Globals.find(*Name);
  if (GI != Globals.end()) {
    TypePlaneMap::iterator TPI = GI->second.begin();
    TypePlaneMap::iterator TPE = GI->second.end();
    for ( ; TPI != TPE ; ++TPI) {
      if (TPI->first->sameNewTyAs(Ty)) 
        return TPI->second;
    }
  }

  if (isGlobal) {
    // We didn't find a global name, but if its supposed to be global then all 
    // we can do is return the name. This is probably a forward reference of a 
    // global value that hasn't been defined yet. Since we have no definition
    // we don't know its linkage class. Just assume its an external and the name
    // shouldn't change.
    return *Name;
  }

  // Remove as many levels of pointer nesting that we have.
  if (Ty->isPointer()) {
    // Avoid infinite loops in recursive types
    const TypeInfo* Last = 0;
    while (Ty->isPointer() && !Ty->sameOldTyAs(Last)) {
      Last = Ty;
      Ty = Ty->getElementType()->resolve();
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
      const TypeInfo* Tmp = Ty->getElement(i);
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

static unsigned UniqueNameCounter = 0;

std::string getGlobalName(const std::string* Name, const std::string Linkage,
                          const TypeInfo* Ty, bool isConstant) {
  // Default to given name
  std::string Result = *Name; 
  // Look up the name in the Globals Map
  GlobalsTypeMap::iterator GI = Globals.find(*Name);
  // Did we see this global name before?
  if (GI != Globals.end()) {
    if (Ty->isUnresolvedDeep()) {
      // The Gval's type is unresolved. Consequently, we can't disambiguate it
      // by type. We'll just change its name and emit a warning.
      warning("Cannot disambiguate global value '" + *Name + 
              "' because type '" + Ty->getNewTy() + "'is unresolved.\n");
      Result = *Name + ".unique";
      UniqueNameCounter++;
      Result += llvm::utostr(UniqueNameCounter);
      return Result;
    } else {
      TypePlaneMap::iterator TPI = GI->second.find(Ty);
      if (TPI != GI->second.end()) {
        // We found an existing name of the same old type. This isn't allowed 
        // in LLVM 2.0. Consequently, we must alter the name of the global so it
        // can at least compile. References to the global will yield the first
        // definition, which is okay. We also must warn about this.
        Result = *Name + ".unique";
        UniqueNameCounter++;
        Result += llvm::utostr(UniqueNameCounter);
        warning(std::string("Global variable '") + *Name + "' was renamed to '"+
                Result + "'");
      } else { 
        // There isn't an existing definition for this name according to the
        // old types. Now search the TypePlanMap for types with the same new
        // name. 
        TypePlaneMap::iterator TPI = GI->second.begin();
        TypePlaneMap::iterator TPE = GI->second.end();
        for ( ; TPI != TPE; ++TPI) {
          if (TPI->first->sameNewTyAs(Ty)) {
            // The new types are the same but the old types are different so 
            // this is a global name collision resulting from type planes 
            // collapsing. 
            if (Linkage == "external" || Linkage == "dllimport" || 
                Linkage == "extern_weak" || Linkage == "") {
              // The linkage of this gval is external so we can't reliably 
              // rename it because it could potentially create a linking 
              // problem.  However, we can't leave the name conflict in the 
              // output either or it won't assemble with LLVM 2.0.  So, all we 
              // can do is rename this one to something unique and emit a 
              // warning about the problem.
              Result = *Name + ".unique";
              UniqueNameCounter++;
              Result += llvm::utostr(UniqueNameCounter);
              warning("Renaming global value '" + *Name + "' to '" + Result + 
                      "' may cause linkage errors.");
              return Result;
            } else {
              // Its linkage is internal and its type is known so we can 
              // disambiguate the name collision successfully based on the type.
              Result = getUniqueName(Name, Ty);
              TPI->second = Result;
              return Result;
            }
          }
        }
        // We didn't find an entry in the type plane with the same new type and
        // the old types differ so this is a new type plane for this global 
        // variable. We just fall through to the logic below which inserts
        // the global.
      }
    }
  }

  // Its a new global name, if it is external we can't change it
  if (isConstant || Linkage == "external" || Linkage == "dllimport" || 
      Linkage == "extern_weak" || Linkage == "") {
    Globals[Result][Ty] = Result;
    return Result;
  }

  // Its a new global name, and it is internal, change the name to make it
  // unique for its type.
  // Result = getUniqueName(Name, Ty);
  Globals[*Name][Ty] = Result;
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
#line 742 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
typedef union YYSTYPE {
  std::string*    String;
  const TypeInfo* Type;
  ValueInfo       Value;
  ConstInfo       Const;
  ValueList*      ValList;
  TypeList*       TypeVec;
} YYSTYPE;
/* Line 196 of yacc.c.  */
#line 1128 "UpgradeParser.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 1140 "UpgradeParser.tab.c"

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
#define YYLAST   1486

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  165
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  75
/* YYNRULES -- Number of rules. */
#define YYNRULES  301
/* YYNRULES -- Number of states. */
#define YYNSTATES  586

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   405

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     154,   155,   163,     2,   152,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     159,   151,   160,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,   156,   153,   158,     2,     2,     2,     2,     2,   164,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     157,     2,     2,   161,     2,   162,     2,     2,     2,     2,
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
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150
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
      99,   101,   103,   105,   107,   109,   111,   113,   115,   117,
     119,   121,   123,   125,   127,   129,   131,   133,   135,   137,
     139,   141,   143,   145,   147,   149,   151,   153,   155,   157,
     159,   161,   164,   165,   167,   169,   171,   173,   175,   177,
     179,   180,   182,   184,   186,   188,   190,   192,   195,   196,
     197,   200,   201,   205,   208,   209,   211,   212,   216,   218,
     221,   223,   225,   227,   229,   231,   233,   235,   237,   239,
     241,   243,   245,   247,   249,   251,   253,   255,   257,   259,
     261,   264,   269,   275,   281,   285,   288,   294,   299,   302,
     304,   308,   310,   314,   316,   317,   322,   326,   330,   335,
     340,   344,   347,   350,   353,   356,   359,   362,   365,   368,
     371,   374,   381,   387,   396,   403,   410,   417,   425,   433,
     440,   447,   456,   465,   469,   471,   473,   475,   477,   480,
     483,   488,   491,   493,   495,   497,   502,   505,   510,   517,
     524,   531,   538,   542,   547,   548,   550,   552,   554,   558,
     562,   566,   570,   574,   578,   580,   581,   583,   585,   587,
     588,   591,   595,   597,   599,   603,   605,   606,   615,   617,
     619,   623,   625,   627,   631,   632,   634,   636,   640,   641,
     643,   645,   647,   649,   651,   653,   655,   657,   659,   663,
     665,   671,   673,   675,   677,   679,   682,   685,   687,   690,
     693,   694,   696,   698,   700,   703,   706,   710,   720,   730,
     739,   754,   756,   758,   765,   771,   774,   781,   789,   791,
     795,   797,   798,   801,   803,   809,   815,   821,   828,   835,
     838,   843,   848,   855,   860,   865,   872,   879,   882,   890,
     892,   895,   896,   898,   899,   903,   910,   914,   921,   924,
     929,   936
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     199,     0,    -1,    19,    -1,    20,    -1,    17,    -1,    18,
      -1,    80,    -1,    81,    -1,    82,    -1,    83,    -1,    84,
      -1,    85,    -1,    86,    -1,    87,    -1,    88,    -1,    89,
      -1,    90,    -1,    91,    -1,    92,    -1,    93,    -1,    94,
      -1,    95,    -1,    96,    -1,    97,    -1,    98,    -1,    99,
      -1,   102,    -1,   103,    -1,   104,    -1,   105,    -1,   106,
      -1,   107,    -1,   118,    -1,   119,    -1,   120,    -1,   121,
      -1,   108,    -1,   109,    -1,   110,    -1,   111,    -1,   112,
      -1,   113,    -1,   114,    -1,   115,    -1,   116,    -1,   117,
      -1,   118,    -1,   119,    -1,   120,    -1,   121,    -1,    25,
      -1,    26,    -1,   130,    -1,   131,    -1,   132,    -1,   133,
      -1,   139,    -1,   140,    -1,   141,    -1,   142,    -1,   143,
      -1,   144,    -1,   145,    -1,   146,    -1,   147,    -1,   148,
      -1,   149,    -1,   150,    -1,   138,    -1,    11,    -1,     9,
      -1,     7,    -1,     5,    -1,    12,    -1,    10,    -1,     8,
      -1,     6,    -1,   175,    -1,   176,    -1,    13,    -1,    14,
      -1,   208,   151,    -1,    -1,    42,    -1,    43,    -1,    44,
      -1,    48,    -1,    45,    -1,    46,    -1,    47,    -1,    -1,
      66,    -1,    67,    -1,    68,    -1,    69,    -1,    70,    -1,
      71,    -1,    65,    18,    -1,    -1,    -1,    57,    18,    -1,
      -1,   152,    57,    18,    -1,    37,    30,    -1,    -1,   184,
      -1,    -1,   152,   187,   186,    -1,   184,    -1,    57,    18,
      -1,   190,    -1,     3,    -1,   192,    -1,     3,    -1,   192,
      -1,     4,    -1,     5,    -1,     6,    -1,     7,    -1,     8,
      -1,     9,    -1,    10,    -1,    11,    -1,    12,    -1,    13,
      -1,    14,    -1,    15,    -1,    16,    -1,   222,    -1,   191,
      -1,   153,    18,    -1,   189,   154,   194,   155,    -1,   156,
      18,   157,   192,   158,    -1,   159,    18,   157,   192,   160,
      -1,   161,   193,   162,    -1,   161,   162,    -1,   159,   161,
     193,   162,   160,    -1,   159,   161,   162,   160,    -1,   192,
     163,    -1,   192,    -1,   193,   152,   192,    -1,   193,    -1,
     193,   152,    40,    -1,    40,    -1,    -1,   190,   156,   197,
     158,    -1,   190,   156,   158,    -1,   190,   164,    30,    -1,
     190,   159,   197,   160,    -1,   190,   161,   197,   162,    -1,
     190,   161,   162,    -1,   190,    22,    -1,   190,    23,    -1,
     190,   222,    -1,   190,   196,    -1,   190,    24,    -1,   175,
     167,    -1,   176,   167,    -1,     4,    25,    -1,     4,    26,
      -1,   178,    21,    -1,   174,   154,   195,    39,   190,   155,
      -1,   127,   154,   195,   237,   155,    -1,   129,   154,   195,
     152,   195,   152,   195,   155,    -1,   168,   154,   195,   152,
     195,   155,    -1,   169,   154,   195,   152,   195,   155,    -1,
     170,   154,   195,   152,   195,   155,    -1,   100,   171,   154,
     195,   152,   195,   155,    -1,   101,   172,   154,   195,   152,
     195,   155,    -1,   173,   154,   195,   152,   195,   155,    -1,
     135,   154,   195,   152,   195,   155,    -1,   136,   154,   195,
     152,   195,   152,   195,   155,    -1,   137,   154,   195,   152,
     195,   152,   195,   155,    -1,   197,   152,   195,    -1,   195,
      -1,    35,    -1,    36,    -1,   200,    -1,   200,   217,    -1,
     200,   219,    -1,   200,    63,    62,   203,    -1,   200,    31,
      -1,   202,    -1,    50,    -1,    58,    -1,   202,   179,    27,
     188,    -1,   202,   219,    -1,   202,    63,    62,   203,    -1,
     202,   179,   180,   198,   195,   186,    -1,   202,   179,   201,
     198,   190,   186,    -1,   202,   179,    45,   198,   190,   186,
      -1,   202,   179,    47,   198,   190,   186,    -1,   202,    51,
     205,    -1,   202,    59,   151,   206,    -1,    -1,    30,    -1,
      56,    -1,    55,    -1,    53,   151,   204,    -1,    54,   151,
      18,    -1,    52,   151,    30,    -1,    72,   151,    30,    -1,
     156,   207,   158,    -1,   207,   152,    30,    -1,    30,    -1,
      -1,    28,    -1,    30,    -1,   208,    -1,    -1,   190,   209,
      -1,   211,   152,   210,    -1,   210,    -1,   211,    -1,   211,
     152,    40,    -1,    40,    -1,    -1,   181,   188,   208,   154,
     212,   155,   185,   182,    -1,    32,    -1,   161,    -1,   180,
     213,   214,    -1,    33,    -1,   162,    -1,   215,   225,   216,
      -1,    -1,    45,    -1,    47,    -1,    34,   218,   213,    -1,
      -1,    64,    -1,    17,    -1,    18,    -1,    21,    -1,    25,
      -1,    26,    -1,    22,    -1,    23,    -1,    24,    -1,   159,
     197,   160,    -1,   196,    -1,    62,   220,    30,   152,    30,
      -1,   166,    -1,   208,    -1,   222,    -1,   221,    -1,   190,
     223,    -1,   225,   226,    -1,   226,    -1,   227,   229,    -1,
     227,   231,    -1,    -1,    29,    -1,    78,    -1,    77,    -1,
      73,   224,    -1,    73,     3,    -1,    74,    15,   223,    -1,
      74,     4,   223,   152,    15,   223,   152,    15,   223,    -1,
      75,   177,   223,   152,    15,   223,   156,   230,   158,    -1,
      75,   177,   223,   152,    15,   223,   156,   158,    -1,   179,
      76,   181,   188,   223,   154,   234,   155,    39,    15,   223,
     228,    15,   223,    -1,   228,    -1,    79,    -1,   230,   177,
     221,   152,    15,   223,    -1,   177,   221,   152,    15,   223,
      -1,   179,   236,    -1,   190,   156,   223,   152,   223,   158,
      -1,   232,   152,   156,   223,   152,   223,   158,    -1,   224,
      -1,   233,   152,   224,    -1,   233,    -1,    -1,    61,    60,
      -1,    60,    -1,   168,   190,   223,   152,   223,    -1,   169,
     190,   223,   152,   223,    -1,   170,   190,   223,   152,   223,
      -1,   100,   171,   190,   223,   152,   223,    -1,   101,   172,
     190,   223,   152,   223,    -1,    49,   224,    -1,   173,   224,
     152,   224,    -1,   174,   224,    39,   190,    -1,   129,   224,
     152,   224,   152,   224,    -1,   134,   224,   152,   190,    -1,
     135,   224,   152,   224,    -1,   136,   224,   152,   224,   152,
     224,    -1,   137,   224,   152,   224,   152,   224,    -1,   128,
     232,    -1,   235,   181,   188,   223,   154,   234,   155,    -1,
     239,    -1,   152,   233,    -1,    -1,    38,    -1,    -1,   122,
     190,   183,    -1,   122,   190,   152,    10,   223,   183,    -1,
     123,   190,   183,    -1,   123,   190,   152,    10,   223,   183,
      -1,   124,   224,    -1,   238,   125,   190,   223,    -1,   238,
     126,   224,   152,   190,   223,    -1,   127,   190,   223,   237,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   807,   807,   807,   808,   808,   812,   812,   812,   812,
     812,   812,   812,   813,   813,   813,   813,   814,   814,   814,
     815,   815,   815,   815,   815,   815,   816,   816,   816,   816,
     816,   816,   816,   816,   816,   816,   817,   817,   817,   817,
     817,   817,   817,   817,   817,   817,   818,   818,   818,   818,
     818,   818,   819,   819,   819,   819,   820,   820,   820,   820,
     820,   820,   820,   821,   821,   821,   821,   821,   821,   826,
     826,   826,   826,   827,   827,   827,   827,   828,   828,   829,
     829,   832,   835,   840,   840,   840,   840,   840,   840,   841,
     842,   845,   845,   845,   845,   845,   846,   847,   852,   857,
     858,   861,   862,   870,   876,   877,   880,   881,   890,   891,
     904,   904,   905,   905,   906,   910,   910,   910,   910,   910,
     910,   910,   911,   911,   911,   911,   911,   913,   916,   919,
     922,   926,   939,   945,   951,   961,   964,   974,   977,   985,
     989,   996,   997,  1002,  1007,  1017,  1023,  1028,  1034,  1040,
    1046,  1051,  1057,  1063,  1070,  1076,  1082,  1088,  1094,  1100,
    1106,  1114,  1128,  1140,  1145,  1151,  1156,  1162,  1167,  1172,
    1180,  1185,  1190,  1200,  1205,  1210,  1210,  1220,  1225,  1228,
    1233,  1237,  1241,  1243,  1243,  1246,  1256,  1261,  1266,  1276,
    1286,  1296,  1306,  1311,  1316,  1321,  1323,  1323,  1326,  1331,
    1338,  1343,  1350,  1357,  1362,  1363,  1371,  1371,  1372,  1372,
    1374,  1383,  1387,  1391,  1394,  1399,  1402,  1405,  1423,  1424,
    1427,  1438,  1439,  1441,  1450,  1451,  1452,  1456,  1469,  1470,
    1473,  1473,  1473,  1473,  1473,  1473,  1473,  1474,  1475,  1480,
    1481,  1490,  1490,  1494,  1499,  1509,  1518,  1521,  1529,  1533,
    1538,  1541,  1547,  1547,  1549,  1554,  1559,  1564,  1572,  1580,
    1587,  1609,  1614,  1620,  1626,  1634,  1652,  1660,  1669,  1673,
    1680,  1681,  1685,  1690,  1693,  1702,  1710,  1719,  1727,  1735,
    1740,  1749,  1777,  1783,  1789,  1796,  1802,  1808,  1814,  1832,
    1837,  1838,  1842,  1843,  1846,  1854,  1863,  1871,  1880,  1886,
    1895,  1904
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
  "SETEQ", "SETNE", "ICMP", "FCMP", "EQ", "NE", "SLT", "SGT", "SLE", "SGE",
  "OEQ", "ONE", "OLT", "OGT", "OLE", "OGE", "ORD", "UNO", "UEQ", "UNE",
  "ULT", "UGT", "ULE", "UGE", "MALLOC", "ALLOCA", "FREE", "LOAD", "STORE",
  "GETELEMENTPTR", "PHI_TOK", "SELECT", "SHL", "SHR", "ASHR", "LSHR",
  "VAARG", "EXTRACTELEMENT", "INSERTELEMENT", "SHUFFLEVECTOR", "CAST",
  "TRUNC", "ZEXT", "SEXT", "FPTRUNC", "FPEXT", "FPTOUI", "FPTOSI",
  "UITOFP", "SITOFP", "PTRTOINT", "INTTOPTR", "BITCAST", "'='", "','",
  "'\\\\'", "'('", "')'", "'['", "'x'", "']'", "'<'", "'>'", "'{'", "'}'",
  "'*'", "'c'", "$accept", "IntVal", "EInt64Val", "ArithmeticOps",
  "LogicalOps", "SetCondOps", "IPredicates", "FPredicates", "ShiftOps",
  "CastOps", "SIntType", "UIntType", "IntType", "FPType", "OptAssign",
  "OptLinkage", "OptCallingConv", "OptAlign", "OptCAlign", "SectionString",
  "OptSection", "GlobalVarAttributes", "GlobalVarAttribute", "TypesV",
  "UpRTypesV", "Types", "PrimType", "UpRTypes", "TypeListI",
  "ArgTypeListI", "ConstVal", "ConstExpr", "ConstVector", "GlobalType",
  "Module", "DefinitionList", "External", "ConstPool", "AsmBlock",
  "BigOrLittle", "TargetDefinition", "LibrariesDefinition", "LibList",
  "Name", "OptName", "ArgVal", "ArgListH", "ArgList", "FunctionHeaderH",
  "BEGIN", "FunctionHeader", "END", "Function", "FnDeclareLinkage",
  "FunctionProto", "OptSideEffect", "ConstValueRef", "SymbolicValueRef",
  "ValueRef", "ResolvedVal", "BasicBlockList", "BasicBlock",
  "InstructionList", "Unwind", "BBTerminatorInst", "JumpTable", "Inst",
  "PHIList", "ValueRefList", "ValueRefListE", "OptTailCall", "InstVal",
  "IndexList", "OptVolatile", "MemoryInst", 0
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
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,    61,    44,    92,    40,    41,    91,   120,    93,    60,
      62,   123,   125,    42,    99
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   165,   166,   166,   167,   167,   168,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   169,   169,   169,
     170,   170,   170,   170,   170,   170,   171,   171,   171,   171,
     171,   171,   171,   171,   171,   171,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   173,   173,   173,   173,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   175,
     175,   175,   175,   176,   176,   176,   176,   177,   177,   178,
     178,   179,   179,   180,   180,   180,   180,   180,   180,   180,
     180,   181,   181,   181,   181,   181,   181,   181,   181,   182,
     182,   183,   183,   184,   185,   185,   186,   186,   187,   187,
     188,   188,   189,   189,   190,   191,   191,   191,   191,   191,
     191,   191,   191,   191,   191,   191,   191,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   192,   193,
     193,   194,   194,   194,   194,   195,   195,   195,   195,   195,
     195,   195,   195,   195,   195,   195,   195,   195,   195,   195,
     195,   196,   196,   196,   196,   196,   196,   196,   196,   196,
     196,   196,   196,   197,   197,   198,   198,   199,   200,   200,
     200,   200,   200,   201,   201,   202,   202,   202,   202,   202,
     202,   202,   202,   202,   202,   203,   204,   204,   205,   205,
     205,   205,   206,   207,   207,   207,   208,   208,   209,   209,
     210,   211,   211,   212,   212,   212,   212,   213,   214,   214,
     215,   216,   216,   217,   218,   218,   218,   219,   220,   220,
     221,   221,   221,   221,   221,   221,   221,   221,   221,   221,
     221,   222,   222,   223,   223,   224,   225,   225,   226,   227,
     227,   227,   228,   228,   229,   229,   229,   229,   229,   229,
     229,   229,   229,   230,   230,   231,   232,   232,   233,   233,
     234,   234,   235,   235,   236,   236,   236,   236,   236,   236,
     236,   236,   236,   236,   236,   236,   236,   236,   236,   236,
     237,   237,   238,   238,   239,   239,   239,   239,   239,   239,
     239,   239
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     0,     1,     1,     1,     1,     1,     1,     1,
       0,     1,     1,     1,     1,     1,     1,     2,     0,     0,
       2,     0,     3,     2,     0,     1,     0,     3,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     4,     5,     5,     3,     2,     5,     4,     2,     1,
       3,     1,     3,     1,     0,     4,     3,     3,     4,     4,
       3,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     6,     5,     8,     6,     6,     6,     7,     7,     6,
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
       1,     0,     2,     1,     5,     5,     5,     6,     6,     2,
       4,     4,     6,     4,     4,     6,     6,     2,     7,     1,
       2,     0,     1,     0,     3,     6,     3,     6,     2,     4,
       6,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned short int yydefact[] =
{
     194,     0,    90,   182,     1,   181,   224,    83,    84,    85,
      87,    88,    89,    86,     0,    98,   250,   178,   179,   206,
     207,     0,     0,     0,    90,     0,   186,   225,   226,    98,
       0,     0,    91,    92,    93,    94,    95,    96,     0,     0,
     251,   250,   247,    82,     0,     0,     0,     0,   192,     0,
       0,     0,     0,     0,   183,   184,     0,     0,    81,   227,
     195,   180,    97,   111,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,   125,   126,   127,     2,     3,     0,
       0,     0,     0,   241,     0,     0,   110,   129,   114,   242,
     128,   218,   219,   220,   221,   222,   223,   246,     0,     0,
       0,   253,   252,   262,   293,   261,   248,   249,     0,     0,
       0,     0,   205,   193,   187,   185,   175,   176,     0,     0,
       0,     0,   130,     0,     0,     0,   113,   135,   139,     0,
       0,   144,   138,   255,     0,   254,     0,     0,    72,    76,
      71,    75,    70,    74,    69,    73,    77,    78,     0,   292,
       0,   273,     0,    98,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,    54,    55,     0,     0,     0,     0,
      68,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,     0,     0,     0,     0,     0,    98,   265,
       0,   289,   200,   197,   196,   198,   199,   201,   204,     0,
     106,   106,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,     0,     0,     0,     0,   106,   106,     0,
       0,     0,     0,     0,   134,   216,   143,   141,     0,   230,
     231,   232,   235,   236,   237,   233,   234,   228,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   239,   244,   243,   245,     0,   256,     0,   279,   272,
       0,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,     0,    50,    51,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,     0,   101,
     101,   298,     0,     0,   287,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   202,
       0,   190,   191,   158,   159,     4,     5,   156,   157,   160,
     151,   152,   155,     0,     0,     0,     0,   154,   153,   188,
     189,   112,   112,   137,     0,   140,   215,   209,   212,   213,
       0,     0,   131,   229,     0,     0,     0,     0,     0,     0,
       0,     0,   174,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   294,     0,   296,   291,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   203,     0,     0,   108,   106,   146,
       0,     0,   150,     0,   147,   132,   133,   136,   208,   210,
       0,   104,   142,     0,     0,     0,   291,     0,     0,     0,
       0,     0,   238,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   301,     0,     0,
       0,   283,   284,     0,     0,     0,     0,     0,   280,   281,
       0,   299,     0,   103,   109,   107,   145,   148,   149,   214,
     211,   105,    99,     0,     0,     0,     0,     0,     0,     0,
       0,   173,     0,     0,     0,     0,     0,     0,     0,   271,
       0,     0,   101,   102,   101,   268,   290,     0,     0,     0,
       0,     0,   274,   275,   276,   271,     0,     0,   217,   240,
       0,     0,   162,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   270,     0,   277,   278,     0,   295,
     297,     0,     0,     0,   282,   285,   286,     0,   300,   100,
       0,     0,     0,   170,     0,     0,   164,   165,   166,   169,
     161,     0,   259,     0,     0,     0,   269,   266,     0,   288,
     167,   168,     0,     0,     0,   257,     0,   258,     0,     0,
     267,   163,   171,   172,     0,     0,     0,     0,     0,     0,
     264,     0,     0,   263,     0,   260
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,    83,   337,   266,   267,   268,   291,   308,   269,   270,
     233,   234,   148,   235,    24,    15,    38,   508,   385,   407,
     472,   331,   408,    84,    85,   236,    87,    88,   129,   248,
     372,   271,   373,   118,     1,     2,    57,     3,    61,   215,
      48,   113,   219,    89,   419,   358,   359,   360,    39,    93,
      16,    96,    17,    29,    18,   364,   272,    90,   274,   495,
      41,    42,    43,   105,   106,   554,   107,   314,   524,   525,
     208,   209,   447,   210,   211
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -513
static const short int yypact[] =
{
    -513,    46,   217,   541,  -513,  -513,    82,  -513,  -513,  -513,
    -513,  -513,  -513,  -513,    16,   111,    25,  -513,  -513,  -513,
    -513,    34,   -55,    61,    26,   -23,  -513,  -513,  -513,   111,
     132,   146,  -513,  -513,  -513,  -513,  -513,  -513,   873,   -26,
    -513,   -18,  -513,    47,    19,    23,    39,    67,  -513,    77,
     132,   873,    81,    81,  -513,  -513,    81,    81,  -513,  -513,
    -513,  -513,  -513,    89,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,   227,
     229,    -9,   506,  -513,   133,    95,  -513,  -513,  -111,  -513,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,   901,    36,
     148,  -513,  -513,  -513,  1336,  -513,  -513,  -513,   222,    76,
     250,   239,   241,  -513,  -513,  -513,  -513,  -513,   933,   933,
     963,   933,  -513,   115,   116,   614,  -513,  -513,  -111,  -104,
     120,   216,  -513,    89,  1134,  -513,  1134,  1134,  -513,  -513,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  1134,  -513,
     933,  -513,   218,   111,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,  -513,  -513,   195,    84,   933,   933,   933,   933,
     933,   933,  -513,  -513,  -513,  -513,   933,   933,   933,   933,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,  -513,   933,   933,   933,   933,   933,   111,  -513,
      11,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,   -93,
     129,   129,   121,   166,   169,   191,   193,   220,   223,   237,
     240,   261,   263,   249,   249,   264,  1045,   129,   129,   933,
     933,   126,   -73,   933,  -513,   701,  -513,   138,   136,  -513,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,   228,   195,    84,
     140,   150,   154,   156,   158,   963,   167,   168,   170,   171,
     172,  -513,  -513,  -513,  -513,   175,  -513,   176,  -513,  -513,
     873,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,   933,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,   933,   177,
     178,  -513,  1134,   164,   179,   182,   183,   184,   186,   188,
    1134,  1134,  1134,   189,   284,   873,   933,   933,   312,  -513,
      -8,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,  -513,   741,   963,   673,   314,  -513,  -513,  -513,
    -513,  -113,   -99,  -513,   185,  -111,  -513,   133,  -513,   194,
     192,   774,  -513,  -513,   318,   198,   205,   963,   963,   963,
     963,   963,  -513,   -58,   963,   963,   963,   963,   963,   345,
     348,  1134,  1134,  1134,    -2,  -513,     9,  -513,   212,  1134,
     209,   933,   933,   933,   933,   933,   214,   215,   219,   933,
     933,  1134,  1134,   221,  -513,   338,   352,  -513,   129,  -513,
     -61,   -57,  -513,   -70,  -513,  -513,  -513,  -513,  -513,  -513,
     833,   337,  -513,   224,   963,   963,   212,   231,   235,   236,
     238,   963,  -513,   242,   243,   244,   245,   342,  1134,  1134,
     247,   251,   252,  1134,   371,  1134,   933,  -513,   253,  1134,
     254,  -513,  -513,   255,   257,  1134,  1134,  1134,  -513,  -513,
     248,  -513,   933,  -513,  -513,  -513,  -513,  -513,  -513,  -513,
    -513,  -513,   335,   363,   258,   262,   260,   963,   963,   963,
     963,  -513,   963,   963,   963,   963,   933,   265,   266,   933,
    1134,  1134,   267,  -513,   267,  -513,   268,  1134,   269,   933,
     933,   933,  -513,  -513,  -513,   933,  1134,   398,  -513,  -513,
     963,   963,  -513,   271,   273,   277,   278,   276,   279,   282,
     283,   286,   403,    15,   268,   289,  -513,  -513,   375,  -513,
    -513,   933,   275,  1134,  -513,  -513,  -513,   292,  -513,  -513,
     294,   295,   963,  -513,   963,   963,  -513,  -513,  -513,  -513,
    -513,  1134,  -513,  1223,    27,   400,  -513,  -513,   297,  -513,
    -513,  -513,   303,   304,   305,  -513,   313,  -513,  1223,   449,
    -513,  -513,  -513,  -513,   451,   316,  1134,  1134,   454,   135,
    -513,  1134,   457,  -513,  1134,  -513
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -513,  -513,   246,   369,   370,   372,   225,   226,   373,   377,
     -98,   -97,  -507,  -513,   436,   458,  -141,  -513,  -303,    63,
    -513,  -220,  -513,   -46,  -513,   -38,  -513,   -68,   -20,  -513,
     130,   256,  -230,    51,  -513,  -513,  -513,  -513,   439,  -513,
    -513,  -513,  -513,     1,  -513,    66,  -513,  -513,   461,  -513,
    -513,  -513,  -513,  -513,   488,  -513,  -512,  -106,    -3,   -88,
    -513,   455,  -513,   -84,  -513,  -513,  -513,  -513,    56,    18,
    -513,  -513,    98,  -513,  -513
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -178
static const short int yytable[] =
{
      86,   332,   146,   147,    25,   115,    91,   387,   443,   124,
     135,    40,   280,    86,   128,    94,   553,   349,   350,   445,
     138,   139,   140,   141,   142,   143,   144,   145,   273,   405,
     273,   273,   138,   139,   140,   141,   142,   143,   144,   145,
     136,   566,   273,  -112,    25,   415,     4,   568,   243,   406,
     132,   137,   132,    51,    40,   444,   575,   128,   244,   328,
     134,   416,   278,   128,   132,   329,   444,   325,     7,     8,
       9,    52,    11,    53,    13,    19,    54,    20,    30,   243,
     220,   221,   431,   238,    55,   130,    44,    45,    46,   354,
     311,   431,   468,   315,   431,   431,    49,   466,   316,   317,
     318,   319,   432,   467,   119,   242,    47,   120,   121,   292,
     293,   247,   134,   410,   411,   413,   116,   117,   323,   324,
      98,    99,   100,    50,   101,   102,   103,    27,    58,    28,
     348,   213,   214,   275,   276,    92,   326,   327,   309,   310,
     134,   312,   313,   134,    95,   277,   333,   334,   134,   134,
     134,   134,   125,   138,   139,   140,   141,   142,   143,   144,
     145,    19,    60,    20,    62,   320,   321,   322,   134,   134,
     108,   351,   352,   552,   109,   355,    31,    32,    33,    34,
      35,    36,    37,   -72,   -72,   567,   -76,   -76,   465,   529,
     110,   530,   294,   295,   296,   297,   298,   299,   300,   301,
     302,   303,   304,   305,   306,   307,   273,   357,   -71,   -71,
     -75,   -75,   101,   102,   273,   273,   273,  -177,   111,   126,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,   112,   381,    77,    78,   -70,   -70,   403,
     -74,   -74,    86,  -113,    19,   122,    20,   123,     5,   131,
     237,     6,   212,   382,   -69,   -69,   246,   -73,   -73,     7,
       8,     9,    10,    11,    12,    13,   335,   336,   216,   217,
     383,   218,   239,   240,   245,   273,   273,   273,   279,   401,
      14,   330,   -79,   273,   -80,   339,   353,    86,   402,   134,
     361,   362,   363,   355,   367,   273,   273,   281,   282,   283,
     284,   285,   286,   450,   368,   452,   453,   454,   369,   388,
     370,   458,   371,   287,   288,   289,   290,   396,   397,   398,
     389,   374,   375,   400,   376,   377,   378,   379,   380,   384,
     386,   390,   273,   273,   391,   392,   393,   273,   394,   273,
     395,   399,   404,   273,   414,   417,   420,   421,   423,   273,
     273,   273,   424,   134,   451,   134,   134,   134,   418,   425,
     438,   134,   459,   439,   446,   449,   455,   456,   463,    79,
     464,   457,    80,   462,   405,    81,   473,    82,   440,   441,
     442,   486,   357,   477,   273,   273,   448,   478,   479,   493,
     480,   273,   507,   509,   482,   483,   484,   485,   460,   461,
     273,   489,   505,   490,   491,   497,   499,   500,   134,   501,
     510,   534,   535,   536,   511,   512,   539,   522,   551,   528,
     531,   533,   523,   542,   506,   146,   147,   273,   543,   544,
     545,   546,   444,   557,   547,   487,   488,   548,   549,   569,
     492,   550,   494,   556,   555,   273,   498,   559,   521,   560,
     561,   134,   502,   503,   504,   570,   146,   147,   571,   572,
     573,   134,   134,   134,   576,   574,   577,   134,   578,   581,
     273,   273,   584,   203,   204,   273,   205,   206,   273,   104,
     338,   207,    56,   365,   471,   366,   470,   526,   527,   114,
      59,    26,   347,   134,   532,   582,    97,   426,   427,   428,
     429,   430,   496,   538,   433,   434,   435,   436,   437,   126,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,   537,   476,    77,    78,     0,     0,     0,
     558,     0,     0,     0,    19,     0,    20,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   565,     0,
       0,     0,     0,     0,   474,   475,     0,     0,     0,     0,
       0,   481,     0,     0,     0,     0,     0,     0,   -82,    19,
       0,    20,     0,   579,   580,     6,   -82,   -82,   583,     0,
       0,   585,     0,   -82,   -82,   -82,   -82,   -82,   -82,   -82,
       0,   -82,    21,     0,     0,     0,     0,     0,     0,   -82,
      22,     0,     0,     0,    23,     0,     0,   513,   514,   515,
     516,     0,   517,   518,   519,   520,     0,   126,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,     0,     0,    77,    78,     0,     0,     0,     0,     0,
     540,   541,    19,     0,    20,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    79,
       0,     0,    80,     0,     0,    81,     0,    82,   127,     0,
       0,     0,   562,     0,   563,   564,   126,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,    75,    76,
       0,     0,    77,    78,     0,     0,     0,     0,     0,     0,
       0,    19,     0,    20,   126,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,     0,     0,
      77,    78,     0,     0,     0,     0,     0,     0,     0,    19,
       0,    20,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   356,     0,     0,   126,   222,   223,   224,   225,   226,
     227,   228,   229,   230,   231,   232,    75,    76,     0,     0,
      77,    78,     0,     0,     0,     0,     0,    79,     0,    19,
      80,    20,     0,    81,     0,    82,   241,   126,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,     0,     0,    77,    78,     0,     0,     0,     0,     0,
       0,     0,    19,     0,    20,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   422,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    79,     0,     0,    80,
       0,     0,    81,     0,    82,   412,   126,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
       0,     0,    77,    78,    79,     0,     0,    80,     0,     0,
      81,    19,    82,    20,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   469,     0,     0,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
       0,     0,    77,    78,    79,     0,     0,    80,     0,   409,
      81,    19,    82,    20,   133,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,     0,     0,
      77,    78,     0,     0,     0,     0,     0,    79,     0,    19,
      80,    20,     0,    81,     0,    82,   126,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
       0,     0,    77,    78,     0,     0,     0,     0,     0,     0,
       0,    19,     0,    20,     0,     0,   126,   222,   223,   224,
     225,   226,   227,   228,   229,   230,   231,   232,    75,    76,
       0,     0,    77,    78,     0,     0,    79,     0,     0,    80,
       0,    19,    81,    20,    82,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    79,     0,     0,    80,
       0,     0,    81,     0,    82,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    79,     0,     0,    80,     0,     0,
      81,     0,    82,     0,    77,    78,     0,   340,   341,   342,
       0,     0,     0,    19,     0,    20,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    79,     0,     0,    80,
       0,     0,    81,     0,    82,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    79,     0,     0,    80,
       0,     0,    81,     0,    82,   154,   155,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,   168,
     169,   170,   171,   172,   173,   258,   259,     0,     0,     0,
       0,   249,   250,    77,    78,   251,   252,   253,   254,   255,
     256,     0,    19,     0,    20,     0,     0,     0,     0,     0,
       0,     0,   260,     0,   261,   182,   183,   184,   185,     0,
     262,   263,   264,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   257,     0,     0,     0,
       0,   343,     0,     0,   344,     0,   345,     0,     0,   346,
       0,     0,     0,     0,   154,   155,   156,   157,   158,   159,
     160,   161,   162,   163,   164,   165,   166,   167,   168,   169,
     170,   171,   172,   173,   258,   259,     0,     0,     0,     0,
     249,   250,     0,     0,   251,   252,   253,   254,   255,   256,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   260,     0,   261,   182,   183,   184,   185,     0,   262,
     263,   264,   190,   191,   192,   193,   194,   195,   196,   197,
     198,   199,   200,   201,   202,   257,     0,     0,     0,     0,
       0,     0,     0,   265,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   258,   259,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     260,     0,   261,   182,   183,   184,   185,     0,   262,   263,
     264,   190,   191,   192,   193,   194,   195,   196,   197,   198,
     199,   200,   201,   202,   149,     0,     0,     0,     0,     0,
       0,     0,   265,     0,     0,   150,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   151,   152,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   153,     0,     0,     0,   154,   155,   156,   157,
     158,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   174,   175,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   176,   177,
     178,     0,     0,   179,   180,   181,   182,   183,   184,   185,
     186,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,   197,   198,   199,   200,   201,   202
};

static const short int yycheck[] =
{
      38,   221,   100,   100,     3,    51,    32,   310,    10,    18,
      98,    29,   153,    51,    82,    33,   523,   237,   238,    10,
       5,     6,     7,     8,     9,    10,    11,    12,   134,    37,
     136,   137,     5,     6,     7,     8,     9,    10,    11,    12,
       4,   553,   148,   154,    43,   158,     0,   554,   152,    57,
     163,    15,   163,    27,    29,    57,   568,   125,   162,   152,
      98,   160,   150,   131,   163,   158,    57,   208,    42,    43,
      44,    45,    46,    47,    48,    28,    50,    30,    62,   152,
     118,   119,   152,   121,    58,    84,    52,    53,    54,   162,
     178,   152,   162,   181,   152,   152,   151,   158,   186,   187,
     188,   189,   160,   160,    53,   125,    72,    56,    57,    25,
      26,   131,   150,   343,   344,   345,    35,    36,   206,   207,
      73,    74,    75,    62,    77,    78,    79,    45,   151,    47,
     236,    55,    56,   136,   137,   161,   125,   126,   176,   177,
     178,   179,   180,   181,   162,   148,    25,    26,   186,   187,
     188,   189,   161,     5,     6,     7,     8,     9,    10,    11,
      12,    28,    30,    30,    18,   203,   204,   205,   206,   207,
     151,   239,   240,   158,   151,   243,    65,    66,    67,    68,
      69,    70,    71,    17,    18,   158,    17,    18,   408,   492,
     151,   494,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   312,   245,    17,    18,
      17,    18,    77,    78,   320,   321,   322,     0,   151,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,   156,   280,    19,    20,    17,    18,   327,
      17,    18,   280,   154,    28,    18,    30,    18,    31,   154,
     120,    34,    30,   291,    17,    18,    40,    17,    18,    42,
      43,    44,    45,    46,    47,    48,    17,    18,    18,    30,
     308,    30,   157,   157,   154,   381,   382,   383,    60,   325,
      63,   152,    21,   389,    21,    21,   160,   325,   326,   327,
     152,   155,    64,   361,   154,   401,   402,   102,   103,   104,
     105,   106,   107,   391,   154,   393,   394,   395,   154,   312,
     154,   399,   154,   118,   119,   120,   121,   320,   321,   322,
     156,   154,   154,    39,   154,   154,   154,   152,   152,   152,
     152,   152,   438,   439,   152,   152,   152,   443,   152,   445,
     152,   152,    30,   449,    30,   160,   152,   155,    30,   455,
     456,   457,   154,   391,   392,   393,   394,   395,   357,   154,
      15,   399,   400,    15,   152,   156,   152,   152,    30,   153,
      18,   152,   156,   152,    37,   159,   152,   161,   381,   382,
     383,    39,   420,   152,   490,   491,   389,   152,   152,    18,
     152,   497,    57,    30,   152,   152,   152,   152,   401,   402,
     506,   154,   154,   152,   152,   152,   152,   152,   446,   152,
     152,   499,   500,   501,   152,   155,    18,   152,    15,   152,
     152,   152,   156,   152,   462,   523,   523,   533,   155,   152,
     152,   155,    57,   158,   155,   438,   439,   155,   155,    39,
     443,   155,   445,   531,   155,   551,   449,   155,   486,   155,
     155,   489,   455,   456,   457,   158,   554,   554,   155,   155,
     155,   499,   500,   501,    15,   152,    15,   505,   152,    15,
     576,   577,    15,   104,   104,   581,   104,   104,   584,    43,
     234,   104,    24,   258,   421,   259,   420,   490,   491,    50,
      29,     3,   236,   531,   497,   579,    41,   367,   368,   369,
     370,   371,   446,   506,   374,   375,   376,   377,   378,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,   505,   426,    19,    20,    -1,    -1,    -1,
     533,    -1,    -1,    -1,    28,    -1,    30,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   551,    -1,
      -1,    -1,    -1,    -1,   424,   425,    -1,    -1,    -1,    -1,
      -1,   431,    -1,    -1,    -1,    -1,    -1,    -1,    27,    28,
      -1,    30,    -1,   576,   577,    34,    35,    36,   581,    -1,
      -1,   584,    -1,    42,    43,    44,    45,    46,    47,    48,
      -1,    50,    51,    -1,    -1,    -1,    -1,    -1,    -1,    58,
      59,    -1,    -1,    -1,    63,    -1,    -1,   477,   478,   479,
     480,    -1,   482,   483,   484,   485,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,
     510,   511,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   153,
      -1,    -1,   156,    -1,    -1,   159,    -1,   161,   162,    -1,
      -1,    -1,   542,    -1,   544,   545,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    28,    -1,    30,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    -1,
      19,    20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    28,
      -1,    30,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    -1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    -1,
      19,    20,    -1,    -1,    -1,    -1,    -1,   153,    -1,    28,
     156,    30,    -1,   159,    -1,   161,   162,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    40,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,   156,
      -1,    -1,   159,    -1,   161,   162,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,   153,    -1,    -1,   156,    -1,    -1,
     159,    28,   161,    30,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    40,    -1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,   153,    -1,    -1,   156,    -1,   158,
     159,    28,   161,    30,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    -1,
      19,    20,    -1,    -1,    -1,    -1,    -1,   153,    -1,    28,
     156,    30,    -1,   159,    -1,   161,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    28,    -1,    30,    -1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    -1,    19,    20,    -1,    -1,   153,    -1,    -1,   156,
      -1,    28,   159,    30,   161,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,   156,
      -1,    -1,   159,    -1,   161,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   153,    -1,    -1,   156,    -1,    -1,
     159,    -1,   161,    -1,    19,    20,    -1,    22,    23,    24,
      -1,    -1,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,   156,
      -1,    -1,   159,    -1,   161,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   153,    -1,    -1,   156,
      -1,    -1,   159,    -1,   161,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,    -1,    -1,    -1,
      -1,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    -1,    28,    -1,    30,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   127,    -1,   129,   130,   131,   132,   133,    -1,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,    62,    -1,    -1,    -1,
      -1,   156,    -1,    -1,   159,    -1,   161,    -1,    -1,   164,
      -1,    -1,    -1,    -1,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,    -1,    -1,    -1,    -1,
      17,    18,    -1,    -1,    21,    22,    23,    24,    25,    26,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   127,    -1,   129,   130,   131,   132,   133,    -1,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,    62,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   159,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,   100,   101,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     127,    -1,   129,   130,   131,   132,   133,    -1,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,    38,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   159,    -1,    -1,    49,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    60,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    76,    -1,    -1,    -1,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   122,   123,
     124,    -1,    -1,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,   199,   200,   202,     0,    31,    34,    42,    43,    44,
      45,    46,    47,    48,    63,   180,   215,   217,   219,    28,
      30,    51,    59,    63,   179,   208,   219,    45,    47,   218,
      62,    65,    66,    67,    68,    69,    70,    71,   181,   213,
      29,   225,   226,   227,    52,    53,    54,    72,   205,   151,
      62,    27,    45,    47,    50,    58,   180,   201,   151,   213,
      30,   203,    18,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    19,    20,   153,
     156,   159,   161,   166,   188,   189,   190,   191,   192,   208,
     222,    32,   161,   214,    33,   162,   216,   226,    73,    74,
      75,    77,    78,    79,   179,   228,   229,   231,   151,   151,
     151,   151,   156,   206,   203,   188,    35,    36,   198,   198,
     198,   198,    18,    18,    18,   161,     3,   162,   192,   193,
     208,   154,   163,     3,   190,   224,     4,    15,     5,     6,
       7,     8,     9,    10,    11,    12,   175,   176,   177,    38,
      49,    60,    61,    76,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   122,   123,   124,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   168,   169,   170,   173,   174,   235,   236,
     238,   239,    30,    55,    56,   204,    18,    30,    30,   207,
     190,   190,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,   175,   176,   178,   190,   195,   190,   157,
     157,   162,   193,   152,   162,   154,    40,   193,   194,    17,
      18,    21,    22,    23,    24,    25,    26,    62,   100,   101,
     127,   129,   135,   136,   137,   159,   168,   169,   170,   173,
     174,   196,   221,   222,   223,   223,   223,   223,   224,    60,
     181,   102,   103,   104,   105,   106,   107,   118,   119,   120,
     121,   171,    25,    26,   108,   109,   110,   111,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   172,   190,
     190,   224,   190,   190,   232,   224,   224,   224,   224,   224,
     190,   190,   190,   224,   224,   181,   125,   126,   152,   158,
     152,   186,   186,    25,    26,    17,    18,   167,   167,    21,
      22,    23,    24,   156,   159,   161,   164,   196,   222,   186,
     186,   192,   192,   160,   162,   192,    40,   190,   210,   211,
     212,   152,   155,    64,   220,   171,   172,   154,   154,   154,
     154,   154,   195,   197,   154,   154,   154,   154,   154,   152,
     152,   188,   190,   190,   152,   183,   152,   183,   223,   156,
     152,   152,   152,   152,   152,   152,   223,   223,   223,   152,
      39,   188,   190,   224,    30,    37,    57,   184,   187,   158,
     197,   197,   162,   197,    30,   158,   160,   160,   208,   209,
     152,   155,    40,    30,   154,   154,   195,   195,   195,   195,
     195,   152,   160,   195,   195,   195,   195,   195,    15,    15,
     223,   223,   223,    10,    57,    10,   152,   237,   223,   156,
     224,   190,   224,   224,   224,   152,   152,   152,   224,   190,
     223,   223,   152,    30,    18,   186,   158,   160,   162,    40,
     210,   184,   185,   152,   195,   195,   237,   152,   152,   152,
     152,   195,   152,   152,   152,   152,    39,   223,   223,   154,
     152,   152,   223,    18,   223,   224,   233,   152,   223,   152,
     152,   152,   223,   223,   223,   154,   190,    57,   182,    30,
     152,   152,   155,   195,   195,   195,   195,   195,   195,   195,
     195,   190,   152,   156,   233,   234,   223,   223,   152,   183,
     183,   152,   223,   152,   224,   224,   224,   234,   223,    18,
     195,   195,   152,   155,   152,   152,   155,   155,   155,   155,
     155,    15,   158,   177,   230,   155,   224,   158,   223,   155,
     155,   155,   195,   195,   195,   223,   221,   158,   177,    39,
     158,   155,   155,   155,   152,   221,    15,    15,   152,   223,
     223,    15,   228,   223,    15,   223
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
        case 81:
#line 832 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 82:
#line 835 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string(""); 
  ;}
    break;

  case 90:
#line 842 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 97:
#line 847 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += *(yyvsp[0].String); 
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String); 
    ;}
    break;

  case 98:
#line 852 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(""); ;}
    break;

  case 99:
#line 857 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 100:
#line 858 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { *(yyvsp[-1].String) += " " + *(yyvsp[0].String); delete (yyvsp[0].String); (yyval.String) = (yyvsp[-1].String); ;}
    break;

  case 101:
#line 861 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 102:
#line 862 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, ", "); 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 103:
#line 870 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 104:
#line 876 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 106:
#line 880 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 107:
#line 881 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
      (yyvsp[-1].String)->insert(0, ", ");
      if (!(yyvsp[0].String)->empty())
        *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 109:
#line 891 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
      *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
      delete (yyvsp[0].String);
      (yyval.String) = (yyvsp[-1].String);
    ;}
    break;

  case 127:
#line 913 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type) = TypeInfo::get(*(yyvsp[0].String), OpaqueTy);
  ;}
    break;

  case 128:
#line 916 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type) = TypeInfo::get(*(yyvsp[0].String), UnresolvedTy);
  ;}
    break;

  case 129:
#line 919 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.Type) = (yyvsp[0].Type); 
  ;}
    break;

  case 130:
#line 922 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Type UpReference
    (yyvsp[0].String)->insert(0, "\\");
    (yyval.Type) = TypeInfo::get(*(yyvsp[0].String), UpRefTy);
  ;}
    break;

  case 131:
#line 926 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {           // Function derived type?
    std::string newTy( (yyvsp[-3].Type)->getNewTy() + "(");
    for (unsigned i = 0; i < (yyvsp[-1].TypeVec)->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      if ((*(yyvsp[-1].TypeVec))[i]->isVoid())
        newTy += "...";
      else
        newTy += (*(yyvsp[-1].TypeVec))[i]->getNewTy();
    }
    newTy += ")";
    (yyval.Type) = TypeInfo::get(newTy, (yyvsp[-3].Type), (yyvsp[-1].TypeVec));
  ;}
    break;

  case 132:
#line 939 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {          // Sized array type?
    uint64_t elems = atoi((yyvsp[-3].String)->c_str());
    (yyvsp[-3].String)->insert(0,"[ ");
    *(yyvsp[-3].String) += " x " + (yyvsp[-1].Type)->getNewTy() + " ]";
    (yyval.Type) = TypeInfo::get(*(yyvsp[-3].String), ArrayTy, (yyvsp[-1].Type), elems);
  ;}
    break;

  case 133:
#line 945 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {          // Packed array type?
    uint64_t elems = atoi((yyvsp[-3].String)->c_str());
    (yyvsp[-3].String)->insert(0,"< ");
    *(yyvsp[-3].String) += " x " + (yyvsp[-1].Type)->getNewTy() + " >";
    (yyval.Type) = TypeInfo::get(*(yyvsp[-3].String), PackedTy, (yyvsp[-1].Type), elems);
  ;}
    break;

  case 134:
#line 951 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                        // Structure type?
    std::string newTy("{");
    for (unsigned i = 0; i < (yyvsp[-1].TypeVec)->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      newTy += (*(yyvsp[-1].TypeVec))[i]->getNewTy();
    }
    newTy += "}";
    (yyval.Type) = TypeInfo::get(newTy, StructTy, (yyvsp[-1].TypeVec));
  ;}
    break;

  case 135:
#line 961 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                                  // Empty structure type?
    (yyval.Type) = TypeInfo::get("{}", StructTy, new TypeList());
  ;}
    break;

  case 136:
#line 964 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                // Packed Structure type?
    std::string newTy("<{");
    for (unsigned i = 0; i < (yyvsp[-2].TypeVec)->size(); ++i) {
      if (i != 0)
        newTy +=  ", ";
      newTy += (*(yyvsp[-2].TypeVec))[i]->getNewTy();
    }
    newTy += "}>";
    (yyval.Type) = TypeInfo::get(newTy, PackedStructTy, (yyvsp[-2].TypeVec));
  ;}
    break;

  case 137:
#line 974 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                          // Empty packed structure type?
    (yyval.Type) = TypeInfo::get("<{}>", PackedStructTy, new TypeList());
  ;}
    break;

  case 138:
#line 977 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                             // Pointer type?
    (yyval.Type) = (yyvsp[-1].Type)->getPointerType();
  ;}
    break;

  case 139:
#line 985 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.TypeVec) = new TypeList();
    (yyval.TypeVec)->push_back((yyvsp[0].Type));
  ;}
    break;

  case 140:
#line 989 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.TypeVec) = (yyvsp[-2].TypeVec);
    (yyval.TypeVec)->push_back((yyvsp[0].Type));
  ;}
    break;

  case 142:
#line 997 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.TypeVec) = (yyvsp[-2].TypeVec);
    (yyval.TypeVec)->push_back(TypeInfo::get("void",VoidTy));
    delete (yyvsp[0].String);
  ;}
    break;

  case 143:
#line 1002 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.TypeVec) = new TypeList();
    (yyval.TypeVec)->push_back(TypeInfo::get("void",VoidTy));
    delete (yyvsp[0].String);
  ;}
    break;

  case 144:
#line 1007 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.TypeVec) = new TypeList();
  ;}
    break;

  case 145:
#line 1017 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-3].Type)->getNewTy());
    *(yyval.Const).cnst += " [ " + *(yyvsp[-1].String) + " ]";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 146:
#line 1023 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-2].Type)->getNewTy());
    *(yyval.Const).cnst += "[ ]";
  ;}
    break;

  case 147:
#line 1028 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-2].Type)->getNewTy());
    *(yyval.Const).cnst += " c" + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 148:
#line 1034 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { // Nonempty unsized arr
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-3].Type)->getNewTy());
    *(yyval.Const).cnst += " < " + *(yyvsp[-1].String) + " >";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 149:
#line 1040 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-3].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-3].Type)->getNewTy());
    *(yyval.Const).cnst += " { " + *(yyvsp[-1].String) + " }";
    delete (yyvsp[-1].String);
  ;}
    break;

  case 150:
#line 1046 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-2].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-2].Type)->getNewTy());
    *(yyval.Const).cnst += " {}";
  ;}
    break;

  case 151:
#line 1051 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst +=  " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 152:
#line 1057 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 153:
#line 1063 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[0].String), (yyvsp[-1].Type)->resolve(), true);
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + Name;
    delete (yyvsp[0].String);
  ;}
    break;

  case 154:
#line 1070 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 155:
#line 1076 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 156:
#line 1082 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {      // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 157:
#line 1088 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {            // integral constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 158:
#line 1094 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                      // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 159:
#line 1100 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                     // Boolean constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 160:
#line 1106 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                   // Float & Double constants
    (yyval.Const).type = (yyvsp[-1].Type);
    (yyval.Const).cnst = new std::string((yyvsp[-1].Type)->getNewTy());
    *(yyval.Const).cnst += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 161:
#line 1114 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string source = *(yyvsp[-3].Const).cnst;
    const TypeInfo* SrcTy = (yyvsp[-3].Const).type->resolve();
    const TypeInfo* DstTy = (yyvsp[-1].Type)->resolve(); 
    if (*(yyvsp[-5].String) == "cast") {
      // Call getCastUpgrade to upgrade the old cast
      (yyval.String) = new std::string(getCastUpgrade(source, SrcTy, DstTy, true));
    } else {
      // Nothing to upgrade, just create the cast constant expr
      (yyval.String) = new std::string(*(yyvsp[-5].String));
      *(yyval.String) += "( " + source + " to " + (yyvsp[-1].Type)->getNewTy() + ")";
    }
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); delete (yyvsp[-2].String);
  ;}
    break;

  case 162:
#line 1128 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
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

  case 163:
#line 1140 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 164:
#line 1145 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* op = getDivRemOpcode(*(yyvsp[-5].String), (yyvsp[-3].Const).type); 
    (yyval.String) = new std::string(op);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
  ;}
    break;

  case 165:
#line 1151 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 166:
#line 1156 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) = getCompareOp(*(yyvsp[-5].String), (yyvsp[-3].Const).type);
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 167:
#line 1162 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6].String) += "(" + *(yyvsp[-5].String) + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 168:
#line 1167 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-6].String) += "(" + *(yyvsp[-5].String) + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-6].String);
  ;}
    break;

  case 169:
#line 1172 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-5].String)->c_str();
    if (*(yyvsp[-5].String) == "shr")
      shiftop = ((yyvsp[-3].Const).type->isUnsigned()) ? "lshr" : "ashr";
    (yyval.String) = new std::string(shiftop);
    *(yyval.String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    delete (yyvsp[-5].String); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
  ;}
    break;

  case 170:
#line 1180 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += "(" + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 171:
#line 1185 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 172:
#line 1190 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-7].String) += "(" + *(yyvsp[-5].Const).cnst + "," + *(yyvsp[-3].Const).cnst + "," + *(yyvsp[-1].Const).cnst + ")";
    (yyvsp[-5].Const).destroy(); (yyvsp[-3].Const).destroy(); (yyvsp[-1].Const).destroy();
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 173:
#line 1200 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].Const).cnst;
    (yyvsp[0].Const).destroy();
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 174:
#line 1205 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(*(yyvsp[0].Const).cnst); (yyvsp[0].Const).destroy(); ;}
    break;

  case 177:
#line 1220 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
;}
    break;

  case 178:
#line 1225 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 179:
#line 1228 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << '\n';
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 180:
#line 1233 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "module asm " << ' ' << *(yyvsp[0].String) << '\n';
    (yyval.String) = 0;
  ;}
    break;

  case 181:
#line 1237 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "implementation\n";
    (yyval.String) = 0;
  ;}
    break;

  case 182:
#line 1241 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = 0; ;}
    break;

  case 184:
#line 1243 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].String); *(yyval.String) = "external"; ;}
    break;

  case 185:
#line 1246 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    EnumeratedTypes.push_back((yyvsp[0].Type));
    if (!(yyvsp[-2].String)->empty()) {
      NamedTypes[*(yyvsp[-2].String)] = (yyvsp[0].Type);
      *O << *(yyvsp[-2].String) << " = ";
    }
    *O << "type " << (yyvsp[0].Type)->getNewTy() << '\n';
    delete (yyvsp[-2].String); delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 186:
#line 1256 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {       // Function prototypes can be in const pool
    *O << *(yyvsp[0].String) << '\n';
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 187:
#line 1261 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {  // Asm blocks can be in the const pool
    *O << *(yyvsp[-2].String) << ' ' << *(yyvsp[-1].String) << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 188:
#line 1266 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty()) {
      std::string Name = getGlobalName((yyvsp[-4].String),*(yyvsp[-3].String), (yyvsp[-1].Const).type->getPointerType(),
                                       *(yyvsp[-2].String) == "constant");
      *O << Name << " = ";
    }
    *O << *(yyvsp[-3].String) << ' ' << *(yyvsp[-2].String) << ' ' << *(yyvsp[-1].Const).cnst << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String); 
    (yyval.String) = 0;
  ;}
    break;

  case 189:
#line 1276 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty()) {
      std::string Name = getGlobalName((yyvsp[-4].String),*(yyvsp[-3].String),(yyvsp[-1].Type)->getPointerType(),
                                       *(yyvsp[-2].String) == "constant");
      *O << Name << " = ";
    }
    *O <<  *(yyvsp[-3].String) << ' ' << *(yyvsp[-2].String) << ' ' << (yyvsp[-1].Type)->getNewTy() << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 190:
#line 1286 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty()) {
      std::string Name = getGlobalName((yyvsp[-4].String),*(yyvsp[-3].String),(yyvsp[-1].Type)->getPointerType(),
                                       *(yyvsp[-2].String) == "constant");
      *O << Name << " = ";
    }
    *O << *(yyvsp[-3].String) << ' ' << *(yyvsp[-2].String) << ' ' << (yyvsp[-1].Type)->getNewTy() << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 191:
#line 1296 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-4].String)->empty()) {
      std::string Name = getGlobalName((yyvsp[-4].String),*(yyvsp[-3].String),(yyvsp[-1].Type)->getPointerType(),
                                       *(yyvsp[-2].String) == "constant");
      *O << Name << " = ";
    }
    *O << *(yyvsp[-3].String) << ' ' << *(yyvsp[-2].String) << ' ' << (yyvsp[-1].Type)->getNewTy() << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-4].String); delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 192:
#line 1306 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    *O << *(yyvsp[-1].String) << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 193:
#line 1311 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[-2].String) << " = " << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 194:
#line 1316 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyval.String) = 0;
  ;}
    break;

  case 198:
#line 1326 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 199:
#line 1331 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    if (*(yyvsp[0].String) == "64")
      SizeOfPointer = 64;
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 200:
#line 1338 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 201:
#line 1343 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " = " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 202:
#line 1350 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1].String)->insert(0, "[ ");
    *(yyvsp[-1].String) += " ]";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 203:
#line 1357 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 205:
#line 1363 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = new std::string();
  ;}
    break;

  case 209:
#line 1372 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 210:
#line 1374 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
  (yyval.String) = new std::string((yyvsp[-1].Type)->getNewTy());
  if (!(yyvsp[0].String)->empty()) {
    std::string Name = getUniqueName((yyvsp[0].String), (yyvsp[-1].Type)->resolve());
    *(yyval.String) += " " + Name;
  }
  delete (yyvsp[0].String);
;}
    break;

  case 211:
#line 1383 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 212:
#line 1387 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 213:
#line 1391 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 214:
#line 1394 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += ", ...";
    (yyval.String) = (yyvsp[-2].String);
    delete (yyvsp[0].String);
  ;}
    break;

  case 215:
#line 1399 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = (yyvsp[0].String);
  ;}
    break;

  case 216:
#line 1402 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 217:
#line 1405 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-7].String)->empty()) {
      *(yyvsp[-7].String) += " ";
    }
    *(yyvsp[-7].String) += (yyvsp[-6].Type)->getNewTy() + " " + *(yyvsp[-5].String) + "(" + *(yyvsp[-3].String) + ")";
    if (!(yyvsp[-1].String)->empty()) {
      *(yyvsp[-7].String) += " " + *(yyvsp[-1].String);
    }
    if (!(yyvsp[0].String)->empty()) {
      *(yyvsp[-7].String) += " " + *(yyvsp[0].String);
    }
    delete (yyvsp[-5].String);
    delete (yyvsp[-3].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-7].String);
  ;}
    break;

  case 218:
#line 1423 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("{"); delete (yyvsp[0].String); ;}
    break;

  case 219:
#line 1424 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string ("{"); ;}
    break;

  case 220:
#line 1427 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "define ";
    if (!(yyvsp[-2].String)->empty()) {
      *O << *(yyvsp[-2].String) << ' ';
    }
    *O << *(yyvsp[-1].String) << ' ' << *(yyvsp[0].String) << '\n';
    delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 221:
#line 1438 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); delete (yyvsp[0].String); ;}
    break;

  case 222:
#line 1439 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string("}"); ;}
    break;

  case 223:
#line 1441 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
  if ((yyvsp[-1].String))
    *O << *(yyvsp[-1].String);
  *O << *(yyvsp[0].String) << "\n\n";
  delete (yyvsp[-2].String); delete (yyvsp[-1].String); delete (yyvsp[0].String);
  (yyval.String) = 0;
;}
    break;

  case 224:
#line 1450 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 227:
#line 1456 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    if (!(yyvsp[-1].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[-1].String);
    *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[-1].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-2].String);
  ;}
    break;

  case 228:
#line 1469 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 238:
#line 1475 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { 
    (yyvsp[-1].String)->insert(0, "<");
    *(yyvsp[-1].String) += ">";
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 240:
#line 1481 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-3].String)->empty()) {
      *(yyvsp[-4].String) += " " + *(yyvsp[-3].String);
    }
    *(yyvsp[-4].String) += " " + *(yyvsp[-2].String) + ", " + *(yyvsp[0].String);
    delete (yyvsp[-3].String); delete (yyvsp[-2].String); delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-4].String);
  ;}
    break;

  case 243:
#line 1494 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).val = (yyvsp[0].String);
    (yyval.Value).constant = false;
    (yyval.Value).type = 0;
  ;}
    break;

  case 244:
#line 1499 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value).val = (yyvsp[0].String);
    (yyval.Value).constant = true;
    (yyval.Value).type = 0;
  ;}
    break;

  case 245:
#line 1509 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-1].Type) = (yyvsp[-1].Type)->resolve();
    std::string Name = getUniqueName((yyvsp[0].Value).val, (yyvsp[-1].Type));
    (yyval.Value) = (yyvsp[0].Value);
    delete (yyval.Value).val;
    (yyval.Value).val = new std::string((yyvsp[-1].Type)->getNewTy() + " " + Name);
    (yyval.Value).type = (yyvsp[-1].Type);
  ;}
    break;

  case 246:
#line 1518 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 247:
#line 1521 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { // Do not allow functions with 0 basic blocks   
    (yyval.String) = 0;
  ;}
    break;

  case 248:
#line 1529 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 249:
#line 1533 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << '\n';
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 250:
#line 1538 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.String) = 0;
  ;}
    break;

  case 251:
#line 1541 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << *(yyvsp[0].String) << '\n';
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 253:
#line 1547 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = (yyvsp[0].String); *(yyval.String) = "unwind"; ;}
    break;

  case 254:
#line 1549 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {              // Return with a result...
    *O << "    " << *(yyvsp[-1].String) << ' ' << *(yyvsp[0].Value).val << '\n';
    delete (yyvsp[-1].String); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 255:
#line 1554 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                                       // Return with no result...
    *O << "    " << *(yyvsp[-1].String) << ' ' << (yyvsp[0].Type)->getNewTy() << '\n';
    delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 256:
#line 1559 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {                         // Unconditional Branch...
    *O << "    " << *(yyvsp[-2].String) << ' ' << (yyvsp[-1].Type)->getNewTy() << ' ' << *(yyvsp[0].Value).val << '\n';
    delete (yyvsp[-2].String); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 257:
#line 1564 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {  
    std::string Name = getUniqueName((yyvsp[-6].Value).val, (yyvsp[-7].Type));
    *O << "    " << *(yyvsp[-8].String) << ' ' << (yyvsp[-7].Type)->getNewTy() << ' ' << Name << ", " 
       << (yyvsp[-4].Type)->getNewTy() << ' ' << *(yyvsp[-3].Value).val << ", " << (yyvsp[-1].Type)->getNewTy() << ' ' 
       << *(yyvsp[0].Value).val << '\n';
    delete (yyvsp[-8].String); (yyvsp[-6].Value).destroy(); (yyvsp[-3].Value).destroy(); (yyvsp[0].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 258:
#line 1572 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[-6].Value).val, (yyvsp[-7].Type));
    *O << "    " << *(yyvsp[-8].String) << ' ' << (yyvsp[-7].Type)->getNewTy() << ' ' << Name << ", " 
       << (yyvsp[-4].Type)->getNewTy() << ' ' << *(yyvsp[-3].Value).val << " [" << *(yyvsp[-1].String) << " ]\n";
    delete (yyvsp[-8].String); (yyvsp[-6].Value).destroy(); (yyvsp[-3].Value).destroy(); 
    delete (yyvsp[-1].String);
    (yyval.String) = 0;
  ;}
    break;

  case 259:
#line 1580 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[-5].Value).val, (yyvsp[-6].Type));
    *O << "    " << *(yyvsp[-7].String) << ' ' << (yyvsp[-6].Type)->getNewTy() << ' ' << Name << ", " 
       << (yyvsp[-3].Type)->getNewTy() << ' ' << *(yyvsp[-2].Value).val << "[]\n";
    delete (yyvsp[-7].String); (yyvsp[-5].Value).destroy(); (yyvsp[-2].Value).destroy();
    (yyval.String) = 0;
  ;}
    break;

  case 260:
#line 1588 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    const TypeInfo* ResTy = getFunctionReturnType((yyvsp[-10].Type));
    *O << "    ";
    if (!(yyvsp[-13].String)->empty()) {
      std::string Name = getUniqueName((yyvsp[-13].String), ResTy);
      *O << Name << " = ";
    }
    *O << *(yyvsp[-12].String) << ' ' << *(yyvsp[-11].String) << ' ' << (yyvsp[-10].Type)->getNewTy() << ' ' << *(yyvsp[-9].Value).val << " (";
    for (unsigned i = 0; i < (yyvsp[-7].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-7].ValList))[i];
      *O << *VI.val;
      if (i+1 < (yyvsp[-7].ValList)->size())
        *O << ", ";
      VI.destroy();
    }
    *O << ") " << *(yyvsp[-5].String) << ' ' << (yyvsp[-4].Type)->getNewTy() << ' ' << *(yyvsp[-3].Value).val << ' ' 
       << *(yyvsp[-2].String) << ' ' << (yyvsp[-1].Type)->getNewTy() << ' ' << *(yyvsp[0].Value).val << '\n';
    delete (yyvsp[-13].String); delete (yyvsp[-12].String); delete (yyvsp[-11].String); (yyvsp[-9].Value).destroy(); delete (yyvsp[-7].ValList); 
    delete (yyvsp[-5].String); (yyvsp[-3].Value).destroy(); delete (yyvsp[-2].String); (yyvsp[0].Value).destroy(); 
    (yyval.String) = 0;
  ;}
    break;

  case 261:
#line 1609 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << '\n';
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 262:
#line 1614 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *O << "    " << *(yyvsp[0].String) << '\n';
    delete (yyvsp[0].String);
    (yyval.String) = 0;
  ;}
    break;

  case 263:
#line 1620 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + (yyvsp[-4].Type)->getNewTy() + " " + *(yyvsp[-3].String) + ", " + (yyvsp[-1].Type)->getNewTy() + " " + 
           *(yyvsp[0].Value).val;
    delete (yyvsp[-3].String); (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-5].String);
  ;}
    break;

  case 264:
#line 1626 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyvsp[-3].String)->insert(0, (yyvsp[-4].Type)->getNewTy() + " " );
    *(yyvsp[-3].String) += ", " + (yyvsp[-1].Type)->getNewTy() + " " + *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    (yyval.String) = (yyvsp[-3].String);
  ;}
    break;

  case 265:
#line 1634 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-1].String)->empty()) {
      // Get a unique name for this value, based on its type.
      std::string Name = getUniqueName((yyvsp[-1].String), (yyvsp[0].Value).type);
      *(yyvsp[-1].String) = Name + " = ";
      if (deleteUselessCastFlag && *deleteUselessCastName == Name) {
        // don't actually delete it, just comment it out
        (yyvsp[-1].String)->insert(0, "; USELSS BITCAST: "); 
        delete deleteUselessCastName;
      }
    }
    *(yyvsp[-1].String) += *(yyvsp[0].Value).val;
    (yyvsp[0].Value).destroy();
    deleteUselessCastFlag = false;
    (yyval.String) = (yyvsp[-1].String); 
  ;}
    break;

  case 266:
#line 1652 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {    // Used for PHI nodes
    std::string Name = getUniqueName((yyvsp[-3].Value).val, (yyvsp[-5].Type));
    Name.insert(0, (yyvsp[-5].Type)->getNewTy() + "[");
    Name += "," + *(yyvsp[-1].Value).val + "]";
    (yyval.Value).val = new std::string(Name);
    (yyval.Value).type = (yyvsp[-5].Type);
    (yyvsp[-3].Value).destroy(); (yyvsp[-1].Value).destroy();
  ;}
    break;

  case 267:
#line 1660 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[-3].Value).val, (yyvsp[-6].Value).type);
    *(yyvsp[-6].Value).val += ", [" + Name + "," + *(yyvsp[-1].Value).val + "]";
    (yyvsp[-3].Value).destroy(); (yyvsp[-1].Value).destroy();
    (yyval.Value) = (yyvsp[-6].Value);
  ;}
    break;

  case 268:
#line 1669 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.ValList) = new ValueList();
    (yyval.ValList)->push_back((yyvsp[0].Value));
  ;}
    break;

  case 269:
#line 1673 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.ValList) = (yyvsp[-2].ValList);
    (yyval.ValList)->push_back((yyvsp[0].Value));
  ;}
    break;

  case 270:
#line 1680 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = (yyvsp[0].ValList); ;}
    break;

  case 271:
#line 1681 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = new ValueList(); ;}
    break;

  case 272:
#line 1685 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].String);
    delete (yyvsp[0].String);
    (yyval.String) = (yyvsp[-1].String);
  ;}
    break;

  case 274:
#line 1693 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* op = getDivRemOpcode(*(yyvsp[-4].String), (yyvsp[-3].Type)); 
    std::string Name1 = getUniqueName((yyvsp[-2].Value).val, (yyvsp[-3].Type));
    std::string Name2 = getUniqueName((yyvsp[0].Value).val, (yyvsp[-3].Type));
    (yyval.Value).val = new std::string(op);
    *(yyval.Value).val += " " + (yyvsp[-3].Type)->getNewTy() + " " + Name1 + ", " + Name2;
    (yyval.Value).type = (yyvsp[-3].Type);
    delete (yyvsp[-4].String); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 275:
#line 1702 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name1 = getUniqueName((yyvsp[-2].Value).val, (yyvsp[-3].Type));
    std::string Name2 = getUniqueName((yyvsp[0].Value).val, (yyvsp[-3].Type));
    *(yyvsp[-4].String) += " " + (yyvsp[-3].Type)->getNewTy() + " " + Name1 + ", " + Name2;
    (yyval.Value).val = (yyvsp[-4].String);
    (yyval.Value).type = (yyvsp[-3].Type);
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 276:
#line 1710 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name1 = getUniqueName((yyvsp[-2].Value).val, (yyvsp[-3].Type));
    std::string Name2 = getUniqueName((yyvsp[0].Value).val, (yyvsp[-3].Type));
    *(yyvsp[-4].String) = getCompareOp(*(yyvsp[-4].String), (yyvsp[-3].Type));
    *(yyvsp[-4].String) += " " + (yyvsp[-3].Type)->getNewTy() + " " + Name1 + ", " + Name2;
    (yyval.Value).val = (yyvsp[-4].String);
    (yyval.Value).type = TypeInfo::get("bool",BoolTy);
    (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 277:
#line 1719 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name1 = getUniqueName((yyvsp[-2].Value).val, (yyvsp[-3].Type));
    std::string Name2 = getUniqueName((yyvsp[0].Value).val, (yyvsp[-3].Type));
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].String) + " " + (yyvsp[-3].Type)->getNewTy() + " " + Name1 + "," + Name2;
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = TypeInfo::get("bool",BoolTy);
    delete (yyvsp[-4].String); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 278:
#line 1727 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name1 = getUniqueName((yyvsp[-2].Value).val, (yyvsp[-3].Type));
    std::string Name2 = getUniqueName((yyvsp[0].Value).val, (yyvsp[-3].Type));
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].String) + " " + (yyvsp[-3].Type)->getNewTy() + " " + Name1 + "," + Name2;
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = TypeInfo::get("bool",BoolTy);
    delete (yyvsp[-4].String); (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 279:
#line 1735 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    (yyval.Value) = (yyvsp[0].Value);
    (yyval.Value).val->insert(0, *(yyvsp[-1].String) + " ");
    delete (yyvsp[-1].String);
  ;}
    break;

  case 280:
#line 1740 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    const char* shiftop = (yyvsp[-3].String)->c_str();
    if (*(yyvsp[-3].String) == "shr")
      shiftop = ((yyvsp[-2].Value).type->isUnsigned()) ? "lshr" : "ashr";
    (yyval.Value).val = new std::string(shiftop);
    *(yyval.Value).val += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyval.Value).type = (yyvsp[-2].Value).type;
    delete (yyvsp[-3].String); delete (yyvsp[-2].Value).val; (yyvsp[0].Value).destroy();
  ;}
    break;

  case 281:
#line 1749 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string source = *(yyvsp[-2].Value).val;
    const TypeInfo* SrcTy = (yyvsp[-2].Value).type->resolve();
    const TypeInfo* DstTy = (yyvsp[0].Type)->resolve();
    (yyval.Value).val = new std::string();
    (yyval.Value).type = DstTy;
    if (*(yyvsp[-3].String) == "cast") {
      *(yyval.Value).val += getCastUpgrade(source, SrcTy, DstTy, false);
    } else {
      *(yyval.Value).val += *(yyvsp[-3].String) + " " + source + " to " + DstTy->getNewTy();
    }
    // Check to see if this is a useless cast of a value to the same name
    // and the same type. Such casts will probably cause redefinition errors
    // when assembled and perform no code gen action so just remove them.
    if (*(yyvsp[-3].String) == "cast" || *(yyvsp[-3].String) == "bitcast")
      if (SrcTy->isInteger() && DstTy->isInteger() &&
          SrcTy->getBitWidth() == DstTy->getBitWidth()) {
        deleteUselessCastFlag = true; // Flag the "Inst" rule
        deleteUselessCastName = new std::string(*(yyvsp[-2].Value).val); // save the name
        size_t pos = deleteUselessCastName->find_first_of("%\"",0);
        if (pos != std::string::npos) {
          // remove the type portion before val
          deleteUselessCastName->erase(0, pos);
        }
      }
    delete (yyvsp[-3].String); (yyvsp[-2].Value).destroy();
    delete (yyvsp[-1].String);
  ;}
    break;

  case 282:
#line 1777 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = (yyvsp[-2].Value).type;
    (yyvsp[-4].Value).destroy(); delete (yyvsp[-2].Value).val; (yyvsp[0].Value).destroy();
  ;}
    break;

  case 283:
#line 1783 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + (yyvsp[0].Type)->getNewTy();
    (yyval.Value).val = (yyvsp[-3].String);
    (yyval.Value).type = (yyvsp[0].Type);
    (yyvsp[-2].Value).destroy();
  ;}
    break;

  case 284:
#line 1789 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-3].String) += " " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyval.Value).val = (yyvsp[-3].String);
    (yyvsp[-2].Value).type = (yyvsp[-2].Value).type->resolve();;
    (yyval.Value).type = (yyvsp[-2].Value).type->getElementType();
    delete (yyvsp[-2].Value).val; (yyvsp[0].Value).destroy();
  ;}
    break;

  case 285:
#line 1796 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = (yyvsp[-4].Value).type;
    delete (yyvsp[-4].Value).val; (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 286:
#line 1802 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-5].String) += " " + *(yyvsp[-4].Value).val + ", " + *(yyvsp[-2].Value).val + ", " + *(yyvsp[0].Value).val;
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = (yyvsp[-4].Value).type;
    delete (yyvsp[-4].Value).val; (yyvsp[-2].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 287:
#line 1808 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyval.Value).val = (yyvsp[-1].String);
    (yyval.Value).type = (yyvsp[0].Value).type;
    delete (yyvsp[0].Value).val;
  ;}
    break;

  case 288:
#line 1814 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-6].String) += " " + *(yyvsp[-5].String);
    if (!(yyvsp[-6].String)->empty())
      *(yyvsp[-6].String) += " ";
    *(yyvsp[-6].String) += (yyvsp[-4].Type)->getNewTy() + " " + *(yyvsp[-3].Value).val + "(";
    for (unsigned i = 0; i < (yyvsp[-1].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[-1].ValList))[i];
      *(yyvsp[-6].String) += *VI.val;
      if (i+1 < (yyvsp[-1].ValList)->size())
        *(yyvsp[-6].String) += ", ";
      VI.destroy();
    }
    *(yyvsp[-6].String) += ")";
    (yyval.Value).val = (yyvsp[-6].String);
    (yyval.Value).type = getFunctionReturnType((yyvsp[-4].Type));
    delete (yyvsp[-5].String); (yyvsp[-3].Value).destroy(); delete (yyvsp[-1].ValList);
  ;}
    break;

  case 290:
#line 1837 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.ValList) = (yyvsp[0].ValList); ;}
    break;

  case 291:
#line 1838 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {  (yyval.ValList) = new ValueList(); ;}
    break;

  case 293:
#line 1843 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    { (yyval.String) = new std::string(); ;}
    break;

  case 294:
#line 1846 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + (yyvsp[-1].Type)->getNewTy();
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyval.Value).val = (yyvsp[-2].String);
    (yyval.Value).type = (yyvsp[-1].Type)->getPointerType();
    delete (yyvsp[0].String);
  ;}
    break;

  case 295:
#line 1854 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[-1].Value).val, (yyvsp[-2].Type));
    *(yyvsp[-5].String) += " " + (yyvsp[-4].Type)->getNewTy() + ", " + (yyvsp[-2].Type)->getNewTy() + " " + Name;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = (yyvsp[-4].Type)->getPointerType();
    (yyvsp[-1].Value).destroy(); delete (yyvsp[0].String);
  ;}
    break;

  case 296:
#line 1863 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-2].String) += " " + (yyvsp[-1].Type)->getNewTy();
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-2].String) += " " + *(yyvsp[0].String);
    (yyval.Value).val = (yyvsp[-2].String);
    (yyval.Value).type = (yyvsp[-1].Type)->getPointerType();
    delete (yyvsp[0].String);
  ;}
    break;

  case 297:
#line 1871 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[-1].Value).val, (yyvsp[-2].Type));
    *(yyvsp[-5].String) += " " + (yyvsp[-4].Type)->getNewTy() + ", " + (yyvsp[-2].Type)->getNewTy() + " " + Name;
    if (!(yyvsp[0].String)->empty())
      *(yyvsp[-5].String) += " " + *(yyvsp[0].String);
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = (yyvsp[-4].Type)->getPointerType();
    (yyvsp[-1].Value).destroy(); delete (yyvsp[0].String);
  ;}
    break;

  case 298:
#line 1880 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    *(yyvsp[-1].String) += " " + *(yyvsp[0].Value).val;
    (yyval.Value).val = (yyvsp[-1].String);
    (yyval.Value).type = TypeInfo::get("void", VoidTy); 
    (yyvsp[0].Value).destroy();
  ;}
    break;

  case 299:
#line 1886 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[0].Value).val, (yyvsp[-1].Type));
    if (!(yyvsp[-3].String)->empty())
      *(yyvsp[-3].String) += " ";
    *(yyvsp[-3].String) += *(yyvsp[-2].String) + " " + (yyvsp[-1].Type)->getNewTy() + " " + Name;
    (yyval.Value).val = (yyvsp[-3].String);
    (yyval.Value).type = (yyvsp[-1].Type)->getElementType();
    delete (yyvsp[-2].String); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 300:
#line 1895 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[0].Value).val, (yyvsp[-1].Type));
    if (!(yyvsp[-5].String)->empty())
      *(yyvsp[-5].String) += " ";
    *(yyvsp[-5].String) += *(yyvsp[-4].String) + " " + *(yyvsp[-3].Value).val + ", " + (yyvsp[-1].Type)->getNewTy() + " " + Name;
    (yyval.Value).val = (yyvsp[-5].String);
    (yyval.Value).type = TypeInfo::get("void", VoidTy);
    delete (yyvsp[-4].String); (yyvsp[-3].Value).destroy(); (yyvsp[0].Value).destroy();
  ;}
    break;

  case 301:
#line 1904 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"
    {
    std::string Name = getUniqueName((yyvsp[-1].Value).val, (yyvsp[-2].Type));
    // Upgrade the indices
    for (unsigned i = 0; i < (yyvsp[0].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[0].ValList))[i];
      if (VI.type->isUnsigned() && !VI.isConstant() && 
          VI.type->getBitWidth() < 64) {
        std::string* old = VI.val;
        *O << "    %gep_upgrade" << unique << " = zext " << *old 
           << " to i64\n";
        VI.val = new std::string("i64 %gep_upgrade" + llvm::utostr(unique++));
        VI.type = TypeInfo::get("i64",ULongTy);
      }
    }
    *(yyvsp[-3].String) += " " + (yyvsp[-2].Type)->getNewTy() + " " + Name;
    for (unsigned i = 0; i < (yyvsp[0].ValList)->size(); ++i) {
      ValueInfo& VI = (*(yyvsp[0].ValList))[i];
      *(yyvsp[-3].String) += ", " + *VI.val;
    }
    (yyval.Value).val = (yyvsp[-3].String);
    (yyval.Value).type = getGEPIndexedType((yyvsp[-2].Type),(yyvsp[0].ValList)); 
    (yyvsp[-1].Value).destroy(); delete (yyvsp[0].ValList);
  ;}
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 4436 "UpgradeParser.tab.c"

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


#line 1928 "/proj/llvm/llvm-3/tools/llvm-upgrade/UpgradeParser.y"


int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = where + "error: " + std::string(ErrorMsg) + 
                       " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
  *O << "llvm-upgrade parse failed.\n";
  exit(1);
}

static void warning(const std::string& ErrorMsg) {
  std::string where 
    = std::string((CurFilename == "-") ? std::string("<stdin>") : CurFilename)
                  + ":" + llvm::utostr((unsigned) Upgradelineno) + ": ";
  std::string errMsg = where + "warning: " + std::string(ErrorMsg) + 
                       " while reading ";
  if (yychar == YYEMPTY || yychar == 0)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Upgradetext, Upgradeleng) + "'";
  std::cerr << "llvm-upgrade: " << errMsg << '\n';
}

