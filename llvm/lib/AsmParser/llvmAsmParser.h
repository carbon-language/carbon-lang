typedef union {
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
#define	ESINT64VAL	257
#define	EUINT64VAL	258
#define	SINTVAL	259
#define	UINTVAL	260
#define	FPVAL	261
#define	VOID	262
#define	BOOL	263
#define	SBYTE	264
#define	UBYTE	265
#define	SHORT	266
#define	USHORT	267
#define	INT	268
#define	UINT	269
#define	LONG	270
#define	ULONG	271
#define	FLOAT	272
#define	DOUBLE	273
#define	TYPE	274
#define	LABEL	275
#define	VAR_ID	276
#define	LABELSTR	277
#define	STRINGCONSTANT	278
#define	IMPLEMENTATION	279
#define	ZEROINITIALIZER	280
#define	TRUETOK	281
#define	FALSETOK	282
#define	BEGINTOK	283
#define	ENDTOK	284
#define	DECLARE	285
#define	GLOBAL	286
#define	CONSTANT	287
#define	VOLATILE	288
#define	TO	289
#define	DOTDOTDOT	290
#define	NULL_TOK	291
#define	UNDEF	292
#define	CONST	293
#define	INTERNAL	294
#define	LINKONCE	295
#define	WEAK	296
#define	APPENDING	297
#define	OPAQUE	298
#define	NOT	299
#define	EXTERNAL	300
#define	TARGET	301
#define	TRIPLE	302
#define	ENDIAN	303
#define	POINTERSIZE	304
#define	LITTLE	305
#define	BIG	306
#define	ALIGN	307
#define	DEPLIBS	308
#define	CALL	309
#define	TAIL	310
#define	CC_TOK	311
#define	CCC_TOK	312
#define	FASTCC_TOK	313
#define	COLDCC_TOK	314
#define	RET	315
#define	BR	316
#define	SWITCH	317
#define	INVOKE	318
#define	UNWIND	319
#define	UNREACHABLE	320
#define	ADD	321
#define	SUB	322
#define	MUL	323
#define	DIV	324
#define	REM	325
#define	AND	326
#define	OR	327
#define	XOR	328
#define	SETLE	329
#define	SETGE	330
#define	SETLT	331
#define	SETGT	332
#define	SETEQ	333
#define	SETNE	334
#define	MALLOC	335
#define	ALLOCA	336
#define	FREE	337
#define	LOAD	338
#define	STORE	339
#define	GETELEMENTPTR	340
#define	PHI_TOK	341
#define	CAST	342
#define	SELECT	343
#define	SHL	344
#define	SHR	345
#define	VAARG	346
#define	VAARG_old	347
#define	VANEXT_old	348


extern YYSTYPE llvmAsmlval;
