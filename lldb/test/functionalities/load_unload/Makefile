CC ?= clang
ifeq "$(ARCH)" ""
	ARCH = x86_64
endif

ifeq "$(OS)" ""
	OS = $(shell uname -s)
endif

CFLAGS ?= -g -O0
CWD := $(shell pwd)

LIB_PREFIX := libloadunload_

ifeq "$(OS)" "Darwin"
	CFLAGS += -arch $(ARCH)
	DS := dsymutil
	LD_FLAGS := -dynamiclib
	LIB_A := $(LIB_PREFIX)a.dylib
	LIB_B := $(LIB_PREFIX)b.dylib
	LIB_C := $(LIB_PREFIX)c.dylib
	LIB_D := $(LIB_PREFIX)d.dylib
	EXEC_PATH := "@executable_path"
	EXEC_PATH_A := -install_name $(EXEC_PATH)/$(LIB_A)
	EXEC_PATH_B := -install_name $(EXEC_PATH)/$(LIB_B)
	EXEC_PATH_C := -install_name $(EXEC_PATH)/$(LIB_C)
	EXEC_PATH_D := -install_name $(CWD)/$(LIB_D)
else
	CFLAGS += -fPIC
	LD_FLAGS := -shared
	LIB_DL := -ldl
	LIB_A := $(LIB_PREFIX)a.so
	LIB_B := $(LIB_PREFIX)b.so
	LIB_C := $(LIB_PREFIX)c.so
	LIB_D := $(LIB_PREFIX)d.so
endif

all: a.out $(LIB_A) $(LIB_B) $(LIB_C) $(LIB_D) hidden/$(LIB_D)

a.out: main.o $(LIB_D)
	$(CC) $(CFLAGS) -o a.out main.o -L. -lloadunload_d $(LIB_DL)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

hidden/$(LIB_D): b.o
	$(CC) $(CFLAGS) $(LD_FLAGS) -o hidden/$(LIB_D) d.o
	if [ "$(OS)" = "Darwin" ]; then dsymutil -o hidden/$(LIB_D).dSYM hidden/$(LIB_D); fi

$(LIB_A): a.o $(LIB_B)
	$(CC) $(CFLAGS) $(LD_FLAGS) $(EXEC_PATH_A) -o $(LIB_A) a.o -L. -lloadunload_b
	if [ "$(OS)" = "Darwin" ]; then dsymutil $(LIB_A); fi

a.o: a.c
	$(CC) $(CFLAGS) -c a.c

$(LIB_B): b.o
	$(CC) $(CFLAGS) $(LD_FLAGS) $(EXEC_PATH_B) -o $(LIB_B) b.o
	if [ "$(OS)" = "Darwin" ]; then dsymutil $(LIB_B); fi

b.o: b.c
	$(CC) $(CFLAGS) -c b.c

$(LIB_C): c.o
	$(CC) $(CFLAGS) $(LD_FLAGS) $(EXEC_PATH_C) -o $(LIB_C) c.o
	if [ "$(OS)" = "Darwin" ]; then dsymutil $(LIB_C); fi

c.o: c.c
	$(CC) $(CFLAGS) -c c.c

$(LIB_D): d.o
	$(CC) $(CFLAGS) $(LD_FLAGS) $(EXEC_PATH_D) -o $(LIB_D) d.o
	if [ "$(OS)" = "Darwin" ]; then dsymutil $(LIB_D); fi

d.o: d.c
	$(CC) $(CFLAGS) -c d.c

clean:
	rm -rf $(wildcard *.o *~ *.dylib *.so a.out *.dSYM hidden/*)
