char rcsid_lex[] = "$Id$";

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "b.h"
#include "fe.h"
#include "gram.tab.h"

static char buf[BUFSIZ];

static int yyline = 1;

typedef int (*ReadFn) ARGS((void));

static char *StrCopy ARGS((char *));
static int code_get ARGS((void));
static int simple_get ARGS((void));
static void ReadCharString ARGS((ReadFn, int));
static void ReadCodeBlock ARGS((void));
static void ReadOldComment ARGS((ReadFn));

static char *
StrCopy(s) char *s;
{
	char *t = (char *)zalloc(strlen(s) + 1);
	strcpy(t,s);
	return t;
}

static int
simple_get()
{
	int ch;
	if ((ch = getchar()) == '\n') {
		yyline++;
	}
	return ch;
}

static int
code_get()
{
	int ch;
	if ((ch = getchar()) == '\n') {
		yyline++;
	}
	if (ch != EOF) {
		fputc(ch, outfile);
	}
	return ch;
}

void
yypurge()
{
	while (code_get() != EOF) ;
}


static void
ReadCharString(rdfn, which) ReadFn rdfn; int which;
{
	int ch;
	int backslash = 0;
	int firstline = yyline;

	while ((ch = rdfn()) != EOF) {
		if (ch == which && !backslash) {
			return;
		}
		if (ch == '\\' && !backslash) {
			backslash = 1;
		} else {
			backslash = 0;
		}
	}
	yyerror1("Unexpected EOF in string on line ");
	fprintf(stderr, "%d\n", firstline);
	exit(1);
}

static void
ReadOldComment(rdfn) ReadFn rdfn;
{
	/* will not work for comments delimiter in string */

	int ch;
	int starred = 0;
	int firstline = yyline;

	while ((ch = rdfn()) != EOF) {
		if (ch == '*') {
			starred = 1;
		} else if (ch == '/' && starred) {
			return;
		} else {
			starred = 0;
		}
	}
	yyerror1("Unexpected EOF in comment on line ");
	fprintf(stderr, "%d\n", firstline);
	exit(1);
}

static void
ReadCodeBlock()
{
	int ch;
	int firstline = yyline;

	while ((ch = getchar()) != EOF) {
		if (ch == '%') {
			ch = getchar();
			if (ch != '}') {
				yyerror("bad %%");
			}
			return;
		}
		fputc(ch, outfile);
		if (ch == '\n') {
			yyline++;
		}
		if (ch == '"' || ch == '\'') {
			ReadCharString(code_get, ch);
		} else if (ch == '/') {
			ch = getchar();
			if (ch == '*') {
				fputc(ch, outfile);
				ReadOldComment(code_get);
				continue;
			} else {
				ungetc(ch, stdin);
			}
		}
	}
	yyerror1("Unclosed block of C code started on line ");
	fprintf(stderr, "%d\n", firstline);
	exit(1);
}

static int done;
void
yyfinished()
{
	done = 1;
}

int
yylex()
{
	int ch;
	char *ptr = buf;

	if (done) return 0;
	while ((ch = getchar()) != EOF) {
		switch (ch) {
		case ' ':
		case '\f':
		case '\t':
			continue;
		case '\n':
			yyline++;
			continue;
		case '(':
		case ')':
		case ',':
		case ':':
		case ';':
		case '=':
			return(ch);
		case '/':
			ch = getchar();
			if (ch == '*') {
				ReadOldComment(simple_get);
				continue;
			} else {
				ungetc(ch, stdin);
				yyerror("illegal char /");
				continue;
			}
		case '%':
			ch = getchar();
			switch (ch) {
			case '%':
				return (K_PPERCENT);
			case '{':
				ReadCodeBlock();
				continue;
			case 's':
			case 'g':
			case 't':
				do {
					if (ptr >= &buf[BUFSIZ]) {
						yyerror("ID too long");
						return(ERROR);
					} else {
						*ptr++ = ch;
					}
					ch = getchar();
				} while (isalpha(ch) || isdigit(ch) || ch == '_');
				ungetc(ch, stdin);
				*ptr = '\0';
				if (!strcmp(buf, "term")) return K_TERM;
				if (!strcmp(buf, "start")) return K_START;
				if (!strcmp(buf, "gram")) return K_GRAM;
				yyerror("illegal character after %%");
				continue;
			default:
				yyerror("illegal character after %%");
				continue;
			}
		default:
			if (isalpha(ch) ) {
				do {
					if (ptr >= &buf[BUFSIZ]) {
						yyerror("ID too long");
						return(ERROR);
					} else {
						*ptr++ = ch;
					}
					ch = getchar();
				} while (isalpha(ch) || isdigit(ch) || ch == '_');
				ungetc(ch, stdin);
				*ptr = '\0';
				yylval.y_string = StrCopy(buf);
				return(ID);
			} 
			if (isdigit(ch)) {
				int val=0;
				do {
					val *= 10;
					val += (ch - '0');
					ch = getchar();
				} while (isdigit(ch));
				ungetc(ch, stdin);
				yylval.y_int = val;
				return(INT);
			}
			yyerror1("illegal char ");
			fprintf(stderr, "(\\%03o)\n", ch);
			exit(1);
		}
	}
	return(0);
}

void yyerror1(const char *str)
{
	fprintf(stderr, "line %d: %s", yyline, str);
}

void
yyerror(const char *str)
{
	yyerror1(str);
	fprintf(stderr, "\n");
	exit(1);
}
