/* $Id$ */

struct binding {
	char	*name;
	int	opnum;
};
typedef struct binding	*Binding;

struct arity {
	int	arity;
	List	bindings;
};
typedef struct arity	*Arity;

struct patternAST {
	struct symbol *sym;
	char	*op;
	List	children;
};
typedef struct patternAST	*PatternAST;

struct ruleAST {
	char			*lhs;
	PatternAST		pat;
	int			erulenum;
	IntList			cost;
	struct rule		*rule;
	struct strTableElement	*kids;
	struct strTableElement	*nts;
};
typedef struct ruleAST	*RuleAST;

typedef enum {
	UNKNOWN,
	OPERATOR,
	NONTERMINAL
} TagType;

struct symbol {
	char	*name;
	TagType	tag;
	union {
		NonTerminal	nt;
		Operator	op;
	} u;
};
typedef struct symbol	*Symbol;

struct strTableElement {
	char *str;
	IntList erulenos;
	char *ename;
};
typedef struct strTableElement	*StrTableElement;

struct strTable {
	List elems;
};
typedef struct strTable	*StrTable;

extern void doGrammarNts ARGS((void));
void makeRuleDescArray ARGS((void));
void makeDeltaCostArray ARGS((void));
void makeStateStringArray ARGS((void));

extern StrTable newStrTable ARGS((void));
extern StrTableElement addString ARGS((StrTable, char *, int, int *));

extern void doSpec ARGS((List, List));
extern Arity newArity ARGS((int, List));
extern Binding newBinding ARGS((char *, int));
extern PatternAST newPatternAST ARGS((char *, List));
extern RuleAST newRuleAST ARGS((char *, PatternAST, int, IntList));
extern Symbol enter ARGS((char *, int *));
extern Symbol newSymbol ARGS((char *));

extern void makeDebug ARGS((void));
extern void makeSimple ARGS((void));
extern void makePlanks ARGS((void));
extern void makeOpLabel ARGS((void));
extern void makeChild ARGS((void));
extern void makeOperators ARGS((void));
extern void makeLabel ARGS((void));
extern void makeString ARGS((void));
extern void makeString ARGS((void));
extern void makeReduce ARGS((void));
extern void makeRuleTable ARGS((void));
extern void makeTables ARGS((void));
extern void makeTreecost ARGS((void));
extern void makePrint ARGS((void));
extern void makeRule ARGS((void));
extern void makeNts ARGS((void));
extern void makeKids ARGS((void));
extern void startBurm ARGS((void));
extern void startOptional ARGS((void));
extern void makePlankLabel ARGS((void));
extern void makeStateLabel ARGS((void));
extern void makeStringArray ARGS((void));
extern void makeNonterminalArray ARGS((void));
extern void makeCostArray ARGS((void));
extern void makeLHSmap ARGS((void));
extern void makeClosureArray ARGS((void));
extern void makeOperatorVector ARGS((void));
extern void endOptional ARGS((void));
extern void reportDiagnostics ARGS((void));
extern void makeNonterminals ARGS((void));
extern int opsOfArity ARGS((int));

extern void yypurge ARGS((void));
extern void yyfinished ARGS((void));

extern void printRepresentative ARGS((FILE *, Item_Set));

extern void dumpRules ARGS((List));
extern void dumpDecls ARGS((List));
extern void dumpRuleAST ARGS((RuleAST));
extern void dumpPatternAST ARGS((PatternAST));
extern void dumpArity ARGS((Arity));
extern void dumpBinding ARGS((Binding));
extern void dumpStrTable ARGS((StrTable));

extern int yylex ARGS((void));
extern int yyparse ARGS((void));

extern int	max_ruleAST;
extern List	ruleASTs;

extern FILE	*outfile;
extern const char *prefix;
extern int 	trimflag;
extern int 	speedflag;
extern int 	grammarflag;
