/* $Id$ */

#define MAX_ARITY	2

typedef int ItemSetNum;
typedef int OperatorNum;
typedef int NonTerminalNum;
typedef int RuleNum;
typedef int ArityNum;
typedef int ERuleNum;

extern NonTerminalNum	last_user_nonterminal;
extern NonTerminalNum	max_nonterminal;
extern RuleNum		max_rule;
extern ERuleNum		max_erule_num;
extern int		max_arity;

#ifdef __STDC__
#define ARGS(x) x
#else
#define ARGS(x) ()
#endif

#ifndef NOLEX
#define DELTAWIDTH	4
typedef short DeltaCost[DELTAWIDTH];
typedef short *DeltaPtr;
extern void ASSIGNCOST ARGS((DeltaPtr, DeltaPtr));
extern void ADDCOST ARGS((DeltaPtr, DeltaPtr));
extern void MINUSCOST ARGS((DeltaPtr, DeltaPtr));
extern void ZEROCOST ARGS((DeltaPtr));
extern int LESSCOST ARGS((DeltaPtr, DeltaPtr));
extern int EQUALCOST ARGS((DeltaPtr, DeltaPtr));
#define PRINCIPLECOST(x)	(x[0])
#else
#define DELTAWIDTH	1
typedef int DeltaCost;
typedef int DeltaPtr;
#define ASSIGNCOST(l, r)	((l) = (r))
#define ADDCOST(l, r)		((l) += (r))
#define MINUSCOST(l, r)		((l) -= (r))
#define ZEROCOST(x)		((x) = 0)
#define LESSCOST(l, r)		((l) < (r))
#define EQUALCOST(l, r)		((l) == (r))
#define PRINCIPLECOST(x)	(x)
#endif /* NOLEX */
#define NODIVERGE(c,state,nt,base)		if (prevent_divergence > 0) CHECKDIVERGE(c,state,nt,base);

struct list {
	void		*x;
	struct list	*next;
};
typedef struct list	*List;

struct intlist {
	int		x;
	struct intlist	*next;
};
typedef struct intlist	*IntList;

struct operator {
	char		*name;
	unsigned int	ref:1;
	OperatorNum	num;
	ItemSetNum	baseNum;
	ItemSetNum	stateCount;
	ArityNum	arity;
	struct table	*table;
};
typedef struct operator	*Operator;

struct nonterminal {
	char		*name;
	NonTerminalNum	num;
	ItemSetNum	baseNum;
	ItemSetNum	ruleCount;
	struct plankMap *pmap;

	struct rule	*sampleRule; /* diagnostic---gives "a" rule that with this lhs */
};
typedef struct nonterminal	*NonTerminal;

struct pattern {
	NonTerminal	normalizer;
	Operator	op;		/* NULL if NonTerm -> NonTerm */
	NonTerminal	children[MAX_ARITY];
};
typedef struct pattern	*Pattern;

struct rule {
	DeltaCost	delta;
	ERuleNum	erulenum;
	RuleNum		num;
	RuleNum		newNum;
	NonTerminal	lhs;
	Pattern		pat;
	unsigned int	used:1;
};
typedef struct rule	*Rule;

struct item {
	DeltaCost	delta;
	Rule		rule;
};
typedef struct item	Item;

typedef short 	*Relevant;	/* relevant non-terminals */

typedef Item	*ItemArray;

struct item_set {	/* indexed by NonTerminal */
	ItemSetNum	num;
	ItemSetNum	newNum;
	Operator	op;
	struct item_set *kids[2];
	struct item_set *representative;
	Relevant	relevant;
	ItemArray	virgin;
	ItemArray	closed;
};
typedef struct item_set	*Item_Set;

#define DIM_MAP_SIZE	(1 << 8)
#define GLOBAL_MAP_SIZE	(1 << 15)

struct mapping {	/* should be a hash table for TS -> int */
	List		*hash;
	int		hash_size;
	int		max_size;
	ItemSetNum	count;
	Item_Set	*set;	/* map: int <-> Item_Set */
};
typedef struct mapping	*Mapping;

struct index_map {
	ItemSetNum	max_size;
	Item_Set	*class;
};
typedef struct index_map	Index_Map;

struct dimension {
	Relevant	relevant;
	Index_Map	index_map;
	Mapping		map;
	ItemSetNum	max_size;
	struct plankMap *pmap;
};
typedef struct dimension	*Dimension;


struct table {
	Operator	op;
	List		rules;
	Relevant	relevant;
	Dimension	dimen[MAX_ARITY];	/* 1 for each dimension */
	Item_Set	*transition;	/* maps local indices to global
						itemsets */
};
typedef struct table	*Table;

struct relation {
	Rule	rule;
	DeltaCost	chain;
	NonTerminalNum	nextchain;
	DeltaCost	sibling;
	int		sibFlag;
	int		sibComputed;
};
typedef struct relation	*Relation;

struct queue {
	List head;
	List tail;
};
typedef struct queue	*Queue;

struct plank {
	char *name;
	List fields;
	int width;
};
typedef struct plank	*Plank;

struct except {
	short index;
	short value;
};
typedef struct except	*Exception;

struct plankMap {
	List exceptions;
	int offset;
	struct stateMap *values;
};
typedef struct plankMap	*PlankMap;

struct stateMap {
	char *fieldname;
	Plank plank;
	int width;
	short *value;
};
typedef struct stateMap	*StateMap;

struct stateMapTable {
	List maps;
};

extern void CHECKDIVERGE ARGS((DeltaPtr, Item_Set, int, int));
extern void zero ARGS((Item_Set));
extern ItemArray newItemArray ARGS((void));
extern ItemArray itemArrayCopy ARGS((ItemArray));
extern Item_Set newItem_Set ARGS((Relevant));
extern void freeItem_Set ARGS((Item_Set));
extern Mapping newMapping ARGS((int));
extern NonTerminal newNonTerminal ARGS((char *));
extern int nonTerminalName ARGS((char *, int));
extern Operator newOperator ARGS((char *, OperatorNum, ArityNum));
extern Pattern newPattern ARGS((Operator));
extern Rule newRule ARGS((DeltaPtr, ERuleNum, NonTerminal, Pattern));
extern List newList ARGS((void *, List));
extern IntList newIntList ARGS((int, IntList));
extern int length ARGS((List));
extern List appendList ARGS((void *, List));
extern Table newTable ARGS((Operator));
extern Queue newQ ARGS((void));
extern void addQ ARGS((Queue, Item_Set));
extern Item_Set popQ ARGS((Queue));
extern int equivSet ARGS((Item_Set, Item_Set));
extern Item_Set decode ARGS((Mapping, ItemSetNum));
extern Item_Set encode ARGS((Mapping, Item_Set, int *));
extern void build ARGS((void));
extern Item_Set *transLval ARGS((Table, int, int));

typedef void *	(*ListFn) ARGS((void *));
extern void foreachList ARGS((ListFn, List));
extern void reveachList ARGS((ListFn, List));

extern void addToTable ARGS((Table, Item_Set));

extern void closure ARGS((Item_Set));
extern void trim ARGS((Item_Set));
extern void findChainRules ARGS((void));
extern void findAllPairs ARGS((void));
extern void addRelevant ARGS((Relevant, NonTerminalNum));

extern void *zalloc ARGS((unsigned int));
extern void zfree ARGS((void *));

extern NonTerminal	start;
extern List		rules;
extern List		chainrules;
extern List		operators;
extern List		leaves;
extern List		nonterminals;
extern List		grammarNts;
extern Queue		globalQ;
extern Mapping		globalMap;
extern int		exceptionTolerance;
extern int		prevent_divergence;
extern int		principleCost;
extern int		lexical;
extern struct rule	stub_rule;
extern Relation 	*allpairs;
extern Item_Set		*sortedStates;
extern Item_Set		errorState;

extern void dumpRelevant ARGS((Relevant));
extern void dumpOperator ARGS((Operator, int));
extern void dumpOperator_s ARGS((Operator));
extern void dumpOperator_l ARGS((Operator));
extern void dumpNonTerminal ARGS((NonTerminal));
extern void dumpRule ARGS((Rule));
extern void dumpRuleList ARGS((List));
extern void dumpItem ARGS((Item *));
extern void dumpItem_Set ARGS((Item_Set));
extern void dumpMapping ARGS((Mapping));
extern void dumpQ ARGS((Queue));
extern void dumpIndex_Map ARGS((Index_Map *));
extern void dumpDimension ARGS((Dimension));
extern void dumpPattern ARGS((Pattern));
extern void dumpTable ARGS((Table, int));
extern void dumpTransition ARGS((Table));
extern void dumpCost ARGS((DeltaCost));
extern void dumpAllPairs ARGS((void));
extern void dumpRelation ARGS((Relation));
extern void dumpSortedStates ARGS((void));
extern void dumpSortedRules ARGS((void));
extern int debugTrim;

#ifdef DEBUG
#define debug(a,b)	if (a) b
#else
#define debug(a,b)
#endif
extern int debugTables;

#define TABLE_INCR	8
#define STATES_INCR	64

#ifdef NDEBUG
#define assert(c) ((void) 0)
#else
#define assert(c) ((void) ((c) || fatal(__FILE__,__LINE__)))
#endif

extern void doStart ARGS((char *));
extern void exit ARGS((int));
extern int fatal ARGS((const char *, int));
extern void yyerror ARGS((const char *));
extern void yyerror1 ARGS((const char *));
