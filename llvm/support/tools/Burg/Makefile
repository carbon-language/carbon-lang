# $Id$

LEVEL = ../..

UNAME := $(shell uname)
include $(LEVEL)/Makefile.config
include $(LEVEL)/Makefile.$(UNAME)

TARGET = burg.$(UNAME)

#CFLAGS	=
#CFLAGS	= -O
#CFLAGS	= -O -DNOLEX
CFLAGS	= -g -DDEBUG
#CFLAGS	= -g -DNOLEX -DDEBUG

SRCS = \
	be.c \
	burs.c \
	closure.c \
	delta.c \
	fe.c \
	item.c \
	lex.c \
	list.c \
	main.c \
	map.c \
	nonterminal.c \
	operator.c \
	pattern.c \
	plank.c \
	queue.c \
	rule.c \
	string.c \
	symtab.c \
	table.c \
	trim.c \
	zalloc.c

BU_OBJS = \
	burs.o \
	closure.o \
	delta.o \
	item.o \
	list.o \
	map.o \
	nonterminal.o \
	operator.o \
	pattern.o \
	queue.o \
	rule.o \
	table.o \
	trim.o \
	zalloc.o

BU_OBJS := $(addprefix $(UNAME)/,$(BU_OBJS))

FE_OBJS = \
	be.o \
	fe.o \
	lex.o \
	main.o \
	plank.o \
	string.o \
	symtab.o \
	y.tab.o

FE_OBJS := $(addprefix $(UNAME)/,$(FE_OBJS))

all: $(TARGET)


$(TARGET): $(BU_OBJS) $(FE_OBJS)
	$(CC) -o $(TARGET) $(CFLAGS) $(BU_OBJS) $(FE_OBJS)

y.tab.c y.tab.h: gram.y
	yacc -d gram.y

clean:
	rm -rf $(UNAME) *.o y.tab.h y.tab.c core burg *.aux *.log *.dvi sample sample.c tmp

$(FE_OBJS):	b.h
$(BU_OBJS):	b.h
$(FE_OBJS):	fe.h

lex.o:	y.tab.h

doc.dvi: doc.tex
	latex doc; latex doc

$(UNAME)/%.o: %.c
	$(VERB) $(CC) -c -g $< -o $@

test: $(TARGET) sample.gr
	./$(TARGET) -I     <sample.gr   >sample.c && cc $(CFLAGS) -o sample sample.c && ./sample
	./$(TARGET) -I      sample.gr   >tmp && cmp tmp sample.c
	./$(TARGET) -I     <sample.gr -o tmp && cmp tmp sample.c
	./$(TARGET) -I      sample.gr -o tmp && cmp tmp sample.c
	./$(TARGET) -I -O0 <sample.gr   >tmp && cmp tmp sample.c
	./$(TARGET) -I -=  <sample.gr   >tmp && cmp tmp sample.c
