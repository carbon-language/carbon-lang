typedef union {
  Module                  *ModuleVal;
  Method                  *MethodVal;
  MethodArgument          *MethArgVal;
  BasicBlock              *BasicBlockVal;
  TerminatorInst          *TermInstVal;
  Instruction             *InstVal;
  ConstPoolVal            *ConstVal;
  const Type              *TypeVal;

  list<MethodArgument*>   *MethodArgList;
  list<Value*>            *ValueList;
  list<const Type*>       *TypeList;
  list<pair<Value*, BasicBlock*> > *PHIList;   // Represent the RHS of PHI node
  list<pair<ConstPoolVal*, BasicBlock*> > *JumpTable;
  vector<ConstPoolVal*>   *ConstVector;

  int64_t                  SInt64Val;
  uint64_t                 UInt64Val;
  int                      SIntVal;
  unsigned                 UIntVal;
  double                   FPVal;

  char                    *StrVal;   // This memory is allocated by strdup!
  ValID                    ValIDVal; // May contain memory allocated by strdup

  Instruction::UnaryOps    UnaryOpVal;
  Instruction::BinaryOps   BinaryOpVal;
  Instruction::TermOps     TermOpVal;
  Instruction::MemoryOps   MemOpVal;
  Instruction::OtherOps    OtherOpVal;
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
#define	STRING	274
#define	TYPE	275
#define	LABEL	276
#define	VAR_ID	277
#define	LABELSTR	278
#define	STRINGCONSTANT	279
#define	IMPLEMENTATION	280
#define	TRUE	281
#define	FALSE	282
#define	BEGINTOK	283
#define	END	284
#define	DECLARE	285
#define	TO	286
#define	RET	287
#define	BR	288
#define	SWITCH	289
#define	NOT	290
#define	ADD	291
#define	SUB	292
#define	MUL	293
#define	DIV	294
#define	REM	295
#define	SETLE	296
#define	SETGE	297
#define	SETLT	298
#define	SETGT	299
#define	SETEQ	300
#define	SETNE	301
#define	MALLOC	302
#define	ALLOCA	303
#define	FREE	304
#define	LOAD	305
#define	STORE	306
#define	GETELEMENTPTR	307
#define	PHI	308
#define	CALL	309
#define	CAST	310
#define	SHL	311
#define	SHR	312


extern YYSTYPE llvmAsmlval;
