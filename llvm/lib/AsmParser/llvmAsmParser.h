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
#define	VOID	261
#define	BOOL	262
#define	SBYTE	263
#define	UBYTE	264
#define	SHORT	265
#define	USHORT	266
#define	INT	267
#define	UINT	268
#define	LONG	269
#define	ULONG	270
#define	FLOAT	271
#define	DOUBLE	272
#define	STRING	273
#define	TYPE	274
#define	LABEL	275
#define	VAR_ID	276
#define	LABELSTR	277
#define	STRINGCONSTANT	278
#define	IMPLEMENTATION	279
#define	TRUE	280
#define	FALSE	281
#define	BEGINTOK	282
#define	END	283
#define	DECLARE	284
#define	TO	285
#define	RET	286
#define	BR	287
#define	SWITCH	288
#define	NOT	289
#define	ADD	290
#define	SUB	291
#define	MUL	292
#define	DIV	293
#define	REM	294
#define	SETLE	295
#define	SETGE	296
#define	SETLT	297
#define	SETGT	298
#define	SETEQ	299
#define	SETNE	300
#define	MALLOC	301
#define	ALLOCA	302
#define	FREE	303
#define	LOAD	304
#define	STORE	305
#define	GETFIELD	306
#define	PUTFIELD	307
#define	PHI	308
#define	CALL	309
#define	CAST	310
#define	SHL	311
#define	SHR	312


extern YYSTYPE llvmAsmlval;
