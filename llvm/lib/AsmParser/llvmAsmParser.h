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
     SECTION = 289,
     VOLATILE = 290,
     TO = 291,
     DOTDOTDOT = 292,
     NULL_TOK = 293,
     UNDEF = 294,
     CONST = 295,
     INTERNAL = 296,
     LINKONCE = 297,
     WEAK = 298,
     APPENDING = 299,
     OPAQUE = 300,
     NOT = 301,
     EXTERNAL = 302,
     TARGET = 303,
     TRIPLE = 304,
     ENDIAN = 305,
     POINTERSIZE = 306,
     LITTLE = 307,
     BIG = 308,
     ALIGN = 309,
     DEPLIBS = 310,
     CALL = 311,
     TAIL = 312,
     CC_TOK = 313,
     CCC_TOK = 314,
     FASTCC_TOK = 315,
     COLDCC_TOK = 316,
     RET = 317,
     BR = 318,
     SWITCH = 319,
     INVOKE = 320,
     UNWIND = 321,
     UNREACHABLE = 322,
     ADD = 323,
     SUB = 324,
     MUL = 325,
     DIV = 326,
     REM = 327,
     AND = 328,
     OR = 329,
     XOR = 330,
     SETLE = 331,
     SETGE = 332,
     SETLT = 333,
     SETGT = 334,
     SETEQ = 335,
     SETNE = 336,
     MALLOC = 337,
     ALLOCA = 338,
     FREE = 339,
     LOAD = 340,
     STORE = 341,
     GETELEMENTPTR = 342,
     PHI_TOK = 343,
     CAST = 344,
     SELECT = 345,
     SHL = 346,
     SHR = 347,
     VAARG = 348,
     EXTRACTELEMENT = 349,
     VAARG_old = 350,
     VANEXT_old = 351
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
#define SECTION 289
#define VOLATILE 290
#define TO 291
#define DOTDOTDOT 292
#define NULL_TOK 293
#define UNDEF 294
#define CONST 295
#define INTERNAL 296
#define LINKONCE 297
#define WEAK 298
#define APPENDING 299
#define OPAQUE 300
#define NOT 301
#define EXTERNAL 302
#define TARGET 303
#define TRIPLE 304
#define ENDIAN 305
#define POINTERSIZE 306
#define LITTLE 307
#define BIG 308
#define ALIGN 309
#define DEPLIBS 310
#define CALL 311
#define TAIL 312
#define CC_TOK 313
#define CCC_TOK 314
#define FASTCC_TOK 315
#define COLDCC_TOK 316
#define RET 317
#define BR 318
#define SWITCH 319
#define INVOKE 320
#define UNWIND 321
#define UNREACHABLE 322
#define ADD 323
#define SUB 324
#define MUL 325
#define DIV 326
#define REM 327
#define AND 328
#define OR 329
#define XOR 330
#define SETLE 331
#define SETGE 332
#define SETLT 333
#define SETGT 334
#define SETEQ 335
#define SETNE 336
#define MALLOC 337
#define ALLOCA 338
#define FREE 339
#define LOAD 340
#define STORE 341
#define GETELEMENTPTR 342
#define PHI_TOK 343
#define CAST 344
#define SELECT 345
#define SHL 346
#define SHR 347
#define VAARG 348
#define EXTRACTELEMENT 349
#define VAARG_old 350
#define VANEXT_old 351




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 878 "/proj/llvm/llvm2/lib/AsmParser/llvmAsmParser.y"
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
/* Line 1275 of yacc.c.  */
#line 269 "llvmAsmParser.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE llvmAsmlval;



