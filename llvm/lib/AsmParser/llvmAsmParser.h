/* A Bison parser, made by GNU Bison 1.875d.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004 Free Software Foundation, Inc.

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
     VOID = 263,
     BOOL = 264,
     SBYTE = 265,
     UBYTE = 266,
     SHORT = 267,
     USHORT = 268,
     INT = 269,
     UINT = 270,
     LONG = 271,
     ULONG = 272,
     FLOAT = 273,
     DOUBLE = 274,
     TYPE = 275,
     LABEL = 276,
     VAR_ID = 277,
     LABELSTR = 278,
     STRINGCONSTANT = 279,
     IMPLEMENTATION = 280,
     ZEROINITIALIZER = 281,
     TRUETOK = 282,
     FALSETOK = 283,
     BEGINTOK = 284,
     ENDTOK = 285,
     DECLARE = 286,
     GLOBAL = 287,
     CONSTANT = 288,
     VOLATILE = 289,
     TO = 290,
     DOTDOTDOT = 291,
     NULL_TOK = 292,
     UNDEF = 293,
     CONST = 294,
     INTERNAL = 295,
     LINKONCE = 296,
     WEAK = 297,
     APPENDING = 298,
     OPAQUE = 299,
     NOT = 300,
     EXTERNAL = 301,
     TARGET = 302,
     TRIPLE = 303,
     ENDIAN = 304,
     POINTERSIZE = 305,
     LITTLE = 306,
     BIG = 307,
     DEPLIBS = 308,
     CALL = 309,
     TAIL = 310,
     CC_TOK = 311,
     CCC_TOK = 312,
     FASTCC_TOK = 313,
     COLDCC_TOK = 314,
     RET = 315,
     BR = 316,
     SWITCH = 317,
     INVOKE = 318,
     UNWIND = 319,
     UNREACHABLE = 320,
     ADD = 321,
     SUB = 322,
     MUL = 323,
     DIV = 324,
     REM = 325,
     AND = 326,
     OR = 327,
     XOR = 328,
     SETLE = 329,
     SETGE = 330,
     SETLT = 331,
     SETGT = 332,
     SETEQ = 333,
     SETNE = 334,
     MALLOC = 335,
     ALLOCA = 336,
     FREE = 337,
     LOAD = 338,
     STORE = 339,
     GETELEMENTPTR = 340,
     PHI_TOK = 341,
     CAST = 342,
     SELECT = 343,
     SHL = 344,
     SHR = 345,
     VAARG = 346,
     VAARG_old = 347,
     VANEXT_old = 348
   };
#endif
#define ESINT64VAL 258
#define EUINT64VAL 259
#define SINTVAL 260
#define UINTVAL 261
#define FPVAL 262
#define VOID 263
#define BOOL 264
#define SBYTE 265
#define UBYTE 266
#define SHORT 267
#define USHORT 268
#define INT 269
#define UINT 270
#define LONG 271
#define ULONG 272
#define FLOAT 273
#define DOUBLE 274
#define TYPE 275
#define LABEL 276
#define VAR_ID 277
#define LABELSTR 278
#define STRINGCONSTANT 279
#define IMPLEMENTATION 280
#define ZEROINITIALIZER 281
#define TRUETOK 282
#define FALSETOK 283
#define BEGINTOK 284
#define ENDTOK 285
#define DECLARE 286
#define GLOBAL 287
#define CONSTANT 288
#define VOLATILE 289
#define TO 290
#define DOTDOTDOT 291
#define NULL_TOK 292
#define UNDEF 293
#define CONST 294
#define INTERNAL 295
#define LINKONCE 296
#define WEAK 297
#define APPENDING 298
#define OPAQUE 299
#define NOT 300
#define EXTERNAL 301
#define TARGET 302
#define TRIPLE 303
#define ENDIAN 304
#define POINTERSIZE 305
#define LITTLE 306
#define BIG 307
#define DEPLIBS 308
#define CALL 309
#define TAIL 310
#define CC_TOK 311
#define CCC_TOK 312
#define FASTCC_TOK 313
#define COLDCC_TOK 314
#define RET 315
#define BR 316
#define SWITCH 317
#define INVOKE 318
#define UNWIND 319
#define UNREACHABLE 320
#define ADD 321
#define SUB 322
#define MUL 323
#define DIV 324
#define REM 325
#define AND 326
#define OR 327
#define XOR 328
#define SETLE 329
#define SETGE 330
#define SETLT 331
#define SETGT 332
#define SETEQ 333
#define SETNE 334
#define MALLOC 335
#define ALLOCA 336
#define FREE 337
#define LOAD 338
#define STORE 339
#define GETELEMENTPTR 340
#define PHI_TOK 341
#define CAST 342
#define SELECT 343
#define SHL 344
#define SHR 345
#define VAARG 346
#define VAARG_old 347
#define VANEXT_old 348




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 866 "/usr/home/llvm/obj/../lib/AsmParser/llvmAsmParser.y"
typedef union YYSTYPE {
  llvm::Module                           *ModuleVal;
  llvm::Function                         *FunctionVal;
  std::pair<llvm::PATypeHolder*, char*>  *ArgVal;
  llvm::BasicBlock                       *BasicBlockVal;
  llvm::TerminatorInst                   *TermInstVal;
  llvm::Instruction                      *InstVal;
  llvm::Constant                         *ConstVal;

  const llvm::Type                       *PrimType;
  llvm::PATypeHolder                     *TypeVal;
  llvm::Value                            *ValueVal;

  std::vector<std::pair<llvm::PATypeHolder*,char*> > *ArgList;
  std::vector<llvm::Value*>              *ValueList;
  std::list<llvm::PATypeHolder>          *TypeList;
  // Represent the RHS of PHI node
  std::list<std::pair<llvm::Value*,
                      llvm::BasicBlock*> > *PHIList;
  std::vector<std::pair<llvm::Constant*, llvm::BasicBlock*> > *JumpTable;
  std::vector<llvm::Constant*>           *ConstVector;

  llvm::GlobalValue::LinkageTypes         Linkage;
  int64_t                           SInt64Val;
  uint64_t                          UInt64Val;
  int                               SIntVal;
  unsigned                          UIntVal;
  double                            FPVal;
  bool                              BoolVal;

  char                             *StrVal;   // This memory is strdup'd!
  llvm::ValID                             ValIDVal; // strdup'd memory maybe!

  llvm::Instruction::BinaryOps            BinaryOpVal;
  llvm::Instruction::TermOps              TermOpVal;
  llvm::Instruction::MemoryOps            MemOpVal;
  llvm::Instruction::OtherOps             OtherOpVal;
  llvm::Module::Endianness                Endianness;
} YYSTYPE;
/* Line 1285 of yacc.c.  */
#line 263 "llvmAsmParser.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE llvmAsmlval;



