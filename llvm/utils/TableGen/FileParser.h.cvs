typedef union {
  std::string*                StrVal;
  int                         IntVal;
  llvm::RecTy*                Ty;
  llvm::Init*                 Initializer;
  std::vector<llvm::Init*>*   FieldList;
  std::vector<unsigned>*      BitList;
  llvm::Record*               Rec;
  SubClassRefTy*              SubClassRef;
  std::vector<SubClassRefTy>* SubClassList;
  std::vector<std::pair<llvm::Init*, std::string> >* DagValueList;
} YYSTYPE;
#define	INT	257
#define	BIT	258
#define	STRING	259
#define	BITS	260
#define	LIST	261
#define	CODE	262
#define	DAG	263
#define	CLASS	264
#define	DEF	265
#define	FIELD	266
#define	LET	267
#define	IN	268
#define	SHLTOK	269
#define	SRATOK	270
#define	SRLTOK	271
#define	INTVAL	272
#define	ID	273
#define	VARNAME	274
#define	STRVAL	275
#define	CODEFRAGMENT	276


extern YYSTYPE Filelval;
